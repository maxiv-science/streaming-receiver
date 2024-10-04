import threading
import time
from copy import copy

import zmq
import json
import numpy as np
import cbor2
import logging
from itertools import count
from threading import Thread
from bitshuffle import compress_lz4, decompress_lz4
from .queuey import Queuey
from .processing import convert_tot, decompress_cbf, unpack_mono12p
from dectris.compression import decompress


logger = logging.getLogger(__name__)


class PilatusPipeline:
    def __init__(self, config):
        self.compress = config.get("compress", True)
        self.rotation = config.get("rotate", False)
        self.mask = config.get("mask", [])
        self.tot = config.get("tot", None)
        if self.tot:
            self.tot_tensor = np.load(self.tot)["tot_to_energy_tensor"]
            print("tot_tensor", self.tot_tensor.shape)
        logger.info(
            "initialised PilatusPipeline with compress:%r rotate:%r mask:%s tot:%s",
            self.compress,
            self.rotation,
            self.mask,
            self.tot,
        )

    def __call__(self, header, parts):
        img = np.empty(header["shape"], dtype=np.int32)
        decompress_cbf(parts[1], img)
        header["compression"] = "none"
        logger.debug("process call with hdr %s", header)

        if self.tot:
            output = np.empty(img.shape, dtype=np.float32)
            convert_tot(img, self.tot_tensor, output)
            img = output
            header["type"] = "float32"
            logger.debug("converted TOT")

        # rotation for the custom cosaxs L shaped pilatus 2M
        if self.rotation:
            img = np.rot90(img, -1)
            img = np.ascontiguousarray(img)
            header["shape"] = header["shape"][::-1]
            logger.debug("rotated image")

        # mask for the custom cosaxs L shaped pilatus 2M, crop after rotate
        if self.mask:
            img[
                self.mask[0][0] : self.mask[0][1], self.mask[1][0] : self.mask[1][1]
            ] = -1
            logger.debug("masked rectangle")

        if self.compress:
            header["compression"] = "bslz4"
            img = compress_lz4(img)
            logger.debug("lz4 compressed image")

        return [header, img]


class OrcaPipeline:
    def __init__(self, config):
        pass
        logger.info("initialised OrcaPipeline")

    def __call__(self, header, parts):
        if header["type"] == "mono12p":
            img = np.empty(header["shape"], dtype=np.uint16)
            unpack_mono12p(parts[1], len(parts[1]), img)
            header["type"] = "uint16"
            logger.debug("unpacked mono12p")
        else:
            img = parts[1]
        return [header, img]


class Detector:
    """Detectors that use the standard streaming format"""

    def __init__(self, pipeline=None):
        self.context = zmq.Context(2)
        self.threads = []
        self.pipeline = pipeline
        self.start_info = {}
        logger.info("initialised detector with pipeline %s", self.pipeline)

    def run(self, config, queue):
        ips = config["dcu_host_purple"]
        self.shutdown_event = threading.Event()
        if isinstance(ips, list):
            # connect to a set of endpoints, make sure that they have unique msg_numbers
            ports = config.get("dcu_port_purple", 9999)
            if isinstance(ports, list):
                if len(ips) != len(ports):
                    raise RuntimeError(
                        f"config must have same number of IPs than ports: {ips} and {ports}"
                    )
            else:
                ports = [ports] * len(ips)
            for ip, port in zip(ips, ports):
                wconfig = copy(config)
                wconfig["dcu_host_purple"] = ip
                wconfig["dcu_port_purple"] = port
                t = Thread(
                    target=self.worker,
                    args=(wconfig, queue, self.shutdown_event),
                    daemon=True,
                )
                t.start()
                self.threads.append(t)
                logger.info("created worker thread with config", wconfig)
        else:
            nworkers = config.get("nworkers", 1)
            for i in range(nworkers):
                t = Thread(
                    target=self.worker,
                    args=(config, queue, self.shutdown_event),
                    daemon=True,
                )
                t.start()
                self.threads.append(t)
            logger.info("created %d worker threads", nworkers)

    def close(self):
        self.shutdown_event.set()
        time.sleep(1)
        for t in self.threads:
            t.join()
        self.context.term()
        # self.context.destroy(linger=1)

    def worker(self, config, queue: Queuey, evt):
        data_pull = self.context.socket(zmq.PULL)
        host = config["dcu_host_purple"]
        port = config.get("dcu_port_purple", 9999)
        data_pull.connect(f"tcp://{host}:{port}")
        logger.info("connected to tcp://%s:%d", host, port)

        poller = zmq.Poller()
        poller.register(data_pull, zmq.POLLIN)

        while True:
            while True:
                socks = dict(poller.poll(timeout=1000))
                if data_pull in socks:
                    parts = data_pull.recv_multipart(copy=False)
                    break
                if evt.is_set():
                    data_pull.close()
                    return
            header = json.loads(parts[0].bytes)
            logger.debug("received frame with header %s from host %s", header, host)
            if header["htype"] == "image":
                header.update(self.start_info)
                if self.pipeline:
                    output = self.pipeline(header, parts)
                else:
                    output = [header, *parts[1:]]
                queue.put(output)
            elif header["htype"] == "header":
                rest = [json.loads(p.bytes) for p in parts[1:]]
                self.start_info = {}
                if len(rest) > 0:
                    if "exptime" in rest[0]:
                        self.start_info["exposure_time"] = rest[0]["exptime"]
                logger.info("got header for series %s with rest %s", header, rest)
                queue.put([header, *rest])
            else:
                rest = [json.loads(p.bytes) for p in parts[1:]]
                logger.info("got header for series %s with rest %s", header, rest)
                queue.put([header, *rest])


class Eiger(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)
        self._msg_number = count(0)
        logger.info("initialised Eiger")
        self.rotate = 0
        self.start_info = {}

    def handle_header(self, header, parts, queue):
        info = json.loads(parts[1].bytes)
        appendix = json.loads(parts[8].bytes)
        self.rotate = appendix.get("rotate", 0)
        meta_header = {
            "htype": "header",
            "msg_number": next(self._msg_number),
            "filename": appendix["filename"],
        }
        keys = [
            "count_time",
            "countrate_correction_applied",
            "countrate_correction_count_cutoff",
            "photon_energy",
            "threshold_energy",
            "flatfield_correction_applied",
            "virtual_pixel_correction_applied",
            "pixel_mask_applied",
            "nimages",
            "ntrigger",
            "trigger_mode",
        ]

        self.start_info = {}
        if header["header_detail"] != "none":
            meta_info = {key: info[key] for key in keys}
            self.start_info["exposure_time"] = meta_info["count_time"]
        else:
            meta_info = {}
        meta_info["rotate"] = self.rotate
        logger.info(
            "processed meta_header: %s and meta_info: %s", meta_header, meta_info
        )
        queue.put([meta_header, meta_info])

    def handle_frame(self, header, parts, queue):
        info = json.loads(parts[1].bytes)
        compression = "bslz4" if "bs" in info["encoding"] else "none"
        data_header = {
            "htype": "image",
            "msg_number": next(self._msg_number),
            "frame": header["frame"],
            "shape": info["shape"][::-1],
            "type": info["type"],
            "compression": compression,
        }
        # update data header with constant info from the start message
        data_header.update(self.start_info)
        logger.debug("handled frame with header %s", data_header)

        if self.rotate:
            img = decompress_lz4(
                parts[2].buffer, data_header["shape"], data_header["type"]
            )
            img = np.rot90(img, self.rotate)
            img = np.ascontiguousarray(img)
            blob = compress_lz4(img)
            # flip shape for odd rotations
            if (self.rotate % 2) == 1:
                data_header["shape"] = data_header["shape"][::-1]
        else:
            blob = parts[2]

        queue.put([data_header, blob])

    def worker(self, config, queue: Queuey, evt):
        data_pull = self.context.socket(zmq.PULL)
        host = config["dcu_host_purple"]
        data_pull.connect(f"tcp://{host}:9999")
        logger.info("zmq connected to tcp://%s:9999 (may not have counterpart)", host)
        poller = zmq.Poller()
        poller.register(data_pull, zmq.POLLIN)

        while True:
            while True:
                socks = dict(poller.poll(timeout=1000))
                if data_pull in socks:
                    parts = data_pull.recv_multipart(copy=False)
                    break
                if evt.is_set():
                    data_pull.close()
                    return
            header = json.loads(parts[0].bytes)
            if header["htype"] == "dimage-1.0":
                self.handle_frame(header, parts, queue)

            elif header["htype"] == "dheader-1.0":
                self.handle_header(header, parts, queue)

            elif header["htype"] == "dseries_end-1.0":
                end_header = {
                    "htype": "series_end",
                    "msg_number": next(self._msg_number),
                }
                logger.info("series end")
                queue.put(
                    [
                        end_header,
                    ]
                )


class Lambda(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)

    def run(self, config, queue):
        self.shutdown_event = threading.Event()
        t = Thread(
            target=self.worker,
            args=(config, queue, self.shutdown_event),
            daemon=True,
        )
        t.start()
        self.threads.append(t)

    def worker(self, config, queue: Queuey, evt):
        data_pull = []
        poller = zmq.Poller()
        for i in range(4):
            sock = self.context.socket(zmq.PULL)
            host = config["dcu_host_purple"][i]
            port = 9010 + i
            sock.connect(f"tcp://{host}:{port}")
            data_pull.append(sock)
            poller.register(sock, zmq.POLLIN)
            logger.info("connected to tcp://%s:%d", host, port)

        last_meta_header = {}

        while True:
            parts = []
            headers = []
            while True:
                socks = dict(poller.poll(timeout=1000))
                if set(socks.keys()) == set(data_pull):
                    for s in data_pull:
                        msgs = s.recv_multipart(copy=False)
                        headers.append(json.loads(msgs[0].bytes))
                        parts.append(msgs)
                    logger.debug("received from all 4 sockets %s", headers)
                    break
                if evt.is_set():
                    for s in data_pull:
                        s.close()
                    return

            logger.debug("merge frames together")
            for m in range(1, 4):
                if (headers[0]["htype"] != headers[m]["htype"]) or (
                    headers[0]["msg_number"] != headers[m]["msg_number"]
                ):
                    raise RuntimeError(
                        "Non matching lambda header messages", headers[0], headers[m]
                    )

            # add data blob of all the modules to the merged message
            if headers[0]["htype"] == "image":
                # add x, z, rotation and full shape to image header
                header = headers[0]
                header.update(last_meta_header)
                logger.debug("received image %s", header)
                merged = [
                    header,
                ]
                for m in range(4):
                    merged.append(parts[m][1])
                queue.put(merged)

            elif headers[0]["htype"] == "header":
                try:
                    meta = {"x": [], "y": [], "rotation": []}
                    for m in range(4):
                        info = json.loads(parts[m][1].bytes)
                        for key in ["x", "y", "rotation"]:
                            meta[key].append(int(info[key]))
                        meta["full_shape"] = info["full_shape"]

                    logger.info(
                        "received header with headers %s and meta: %s", headers[0], meta
                    )
                    last_meta_header = meta
                    queue.put([headers[0], meta])
                except Exception as e:
                    logger.error("failed to process series header %s", e.__repr__())

            elif headers[0]["htype"] == "series_end":
                queue.put(
                    [
                        headers[0],
                    ]
                )
                logger.info("series end %s", headers[0])


def decode_multi_dim_array(tag):
    shape, contents = tag.value
    # tuple of shape, dtype, blob
    return shape, *contents


def decode_compression(tag):
    algorithm, elem_size, encoded = tag.value
    return encoded


tag_decoders = {
    40: lambda tag: decode_multi_dim_array(tag),
    64: lambda tag: (np.uint8, tag.value),
    68: lambda tag: (np.uint8, tag.value),
    69: lambda tag: (np.uint16, tag.value),
    70: lambda tag: (np.uint32, tag.value),
    85: lambda tag: (np.float32, tag.value),
    56500: lambda tag: decode_compression(tag),
}


def tag_hook(decoder, tag):
    # print(tag.tag)
    tag_decoder = tag_decoders.get(tag.tag)
    return tag_decoder(tag) if tag_decoder else tag


class DectrisStream2(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)
        self._msg_number = count(0)

    def worker(self, config, queue: Queuey, evt):
        data_pull = self.context.socket(zmq.PULL)
        host = config["dcu_host_purple"]
        data_pull.connect(f"tcp://{host}:31001")
        logger.info("connected to tcp://%s:31001", host)
        poller = zmq.Poller()
        poller.register(data_pull, zmq.POLLIN)

        while True:
            while True:
                socks = dict(poller.poll(timeout=1000))
                if data_pull in socks:
                    msg = data_pull.recv(copy=False)
                    break
                if evt.is_set():
                    data_pull.close()
                    return
            logger.debug("got new message %d", len(msg.bytes))
            msg = cbor2.loads(msg.buffer, tag_hook=tag_hook)
            if msg["type"] == "start":
                logger.info("start of series with user data: %s", msg["user_data"])
                filename = json.loads(msg["user_data"])["filename"]
                meta_header = {
                    "htype": "header",
                    "msg_number": next(self._msg_number),
                    "filename": filename,
                }
                keys = [
                    "count_time",
                    "countrate_correction_enabled",
                    "flatfield_enabled",
                    "virtual_pixel_interpolation_enabled",
                    "pixel_mask_enabled",
                    "number_of_images",
                ]

                meta_info = {key: msg[key] for key in keys if key in msg}
                meta_info.update(msg["threshold_energy"])
                logger.info(
                    "received start, meta_header %s and meta_info %s",
                    meta_header,
                    meta_info,
                )
                queue.put([meta_header, meta_info])

            elif msg["type"] == "image":
                nthresh = len(msg["data"])
                compression = "bslz4"  # if 'bs' in info['encoding'] else 'none'
                out = []
                for i in range(nthresh):
                    data = msg["data"][f"threshold_{i+1}"]
                    shape, dtype, blob = data

                    if not out:
                        dtype = dtype([]).dtype.name
                        data_header = {
                            "htype": "image",
                            "msg_number": next(self._msg_number),
                            "frame": msg["image_id"],
                            "shape": shape,
                            "type": dtype,
                            "compression": compression,
                        }
                        if nthresh > 1:
                            data_header["shape"] = (nthresh, *shape)
                            data_header["chunks"] = (1, *shape)

                        out.append(data_header)
                        logger.debug("processed image with header %s", data_header)

                    out.append(blob)

                queue.put(out)

            elif msg["type"] == "end":
                end_header = {
                    "htype": "series_end",
                    "msg_number": next(self._msg_number),
                }
                queue.put(
                    [
                        end_header,
                    ]
                )
                logger.info("series end %s", end_header)


def og_decode_multi_dim_array(tag, column_major):
    dimensions, contents = tag.value
    if isinstance(contents, list):
        array = np.empty((len(contents),), dtype=object)
        array[:] = contents
    elif isinstance(contents, (np.ndarray, np.generic)):
        array = contents
    else:
        raise cbor2.CBORDecodeValueError("expected array or typed array")
    return array.reshape(dimensions, order="F" if column_major else "C")


def og_decode_typed_array(tag, dtype):
    if not isinstance(tag.value, bytes):
        raise cbor2.CBORDecodeValueError("expected byte string in typed array")
    return np.frombuffer(tag.value, dtype=dtype)


def og_decode_dectris_compression(tag):
    algorithm, elem_size, encoded = tag.value
    return decompress(encoded, algorithm, elem_size=elem_size)


og_tag_decoders = {
    40: lambda tag: og_decode_multi_dim_array(tag, column_major=False),
    64: lambda tag: og_decode_typed_array(tag, dtype="u1"),
    65: lambda tag: og_decode_typed_array(tag, dtype=">u2"),
    66: lambda tag: og_decode_typed_array(tag, dtype=">u4"),
    67: lambda tag: og_decode_typed_array(tag, dtype=">u8"),
    68: lambda tag: og_decode_typed_array(tag, dtype="u1"),
    69: lambda tag: og_decode_typed_array(tag, dtype="<u2"),
    70: lambda tag: og_decode_typed_array(tag, dtype="<u4"),
    71: lambda tag: og_decode_typed_array(tag, dtype="<u8"),
    72: lambda tag: og_decode_typed_array(tag, dtype="i1"),
    73: lambda tag: og_decode_typed_array(tag, dtype=">i2"),
    74: lambda tag: og_decode_typed_array(tag, dtype=">i4"),
    75: lambda tag: og_decode_typed_array(tag, dtype=">i8"),
    77: lambda tag: og_decode_typed_array(tag, dtype="<i2"),
    78: lambda tag: og_decode_typed_array(tag, dtype="<i4"),
    79: lambda tag: og_decode_typed_array(tag, dtype="<i8"),
    80: lambda tag: og_decode_typed_array(tag, dtype=">f2"),
    81: lambda tag: og_decode_typed_array(tag, dtype=">f4"),
    82: lambda tag: og_decode_typed_array(tag, dtype=">f8"),
    83: lambda tag: og_decode_typed_array(tag, dtype=">f16"),
    84: lambda tag: og_decode_typed_array(tag, dtype="<f2"),
    85: lambda tag: og_decode_typed_array(tag, dtype="<f4"),
    86: lambda tag: og_decode_typed_array(tag, dtype="<f8"),
    87: lambda tag: og_decode_typed_array(tag, dtype="<f16"),
    1040: lambda tag: og_decode_multi_dim_array(tag, column_major=True),
    56500: lambda tag: og_decode_dectris_compression(tag),
}


def og_tag_hook(decoder, tag):
    tag_decoder = og_tag_decoders.get(tag.tag)
    return tag_decoder(tag) if tag_decoder else tag


class Jungfrau(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)

    def worker(self, config, queue: Queuey, evt):
        data_pull = self.context.socket(zmq.SUB)
        host = config["dcu_host_purple"]
        port = config.get("dcu_port_purple", 2345)
        data_pull.connect(f"tcp://{host}:{port}")
        data_pull.setsockopt(zmq.SUBSCRIBE, b"")
        logger.info("connected to tcp://%s:%s", host, port)

        self._msg_number = count(0)
        meta_header = {
            "htype": "header",
            "msg_number": next(self._msg_number),
            "filename": "",
        }
        queue.put([meta_header])
        while True:
            message = data_pull.recv()
            message = cbor2.loads(message, tag_hook=og_tag_hook)
            data_header = {
                "htype": "image",
                "msg_number": next(self._msg_number),
                "frame": message["image_id"],
                "shape": message["data"]["default"].shape,
                "type": "int16",
                "spots": message["spots"],
                "compression": "none",
            }
            # update data header with constant info from the start message
            logger.debug("handled frame with header %s", data_header)

            blob = message["data"]["default"].tobytes()

            queue.put([data_header, blob])


class PsiEiger(Detector):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)
        self._msg_number = count(0)
        logger.info("initialised PsiEiger")
        self.start_info = {}
        self.ports = [30001, 30002]
        self.cache = [{} for _ in self.ports]

    def worker(self, config, queue: Queuey, evt):
        data_pull = [self.context.socket(zmq.SUB) for _ in self.ports]
        host = config["dcu_host_purple"]
        poller = zmq.Poller()
        for sock, port in zip(data_pull, self.ports):
            sock.connect(f"tcp://{host}:{port}")
            logger.info("connected to tcp://%s:%d", host, port)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            poller.register(sock, zmq.POLLIN)

        last = [0, 0]
        prelast = {}
        hist = [{}, {}]

        in_scan = False

        while True:
            socks = dict(poller.poll())
            for sock, cache, i in zip(data_pull, self.cache, [0, 1]):
                if sock in socks and socks[sock] == zmq.POLLIN:
                    parts = sock.recv_multipart(copy=False)
                    header = json.loads(parts[0].bytes)
                    if header["frameNumber"] == 1:
                        hist = [{}, {}]
                    if header["frameNumber"] != 0:
                        hist[i][header["frameNumber"] * 10] = parts
                        prelast[i] = header["frameNumber"] * 10
                    else:
                        hist[i][last[i] + 1] = parts

            common = set(hist[0].keys()).intersection(set(hist[1].keys()))
            print("common", common)
            for c in common:
                d = []
                for i in range(2):
                    d.append(hist[i].pop(c))

                print("sendout", d)

                header = json.loads(d[0][0].bytes)
                if header["size"] == 0:
                    end_header = {
                        "htype": "series_end",
                        "msg_number": next(self._msg_number),
                    }
                    logger.info("series end")
                    queue.put(
                        [
                            end_header,
                        ]
                    )
                    in_scan = False
                    logger.debug("started, in_scan set to %s", in_scan)
                else:
                    if in_scan is False:
                        meta_header = {
                            "htype": "header",
                            "msg_number": next(self._msg_number),
                            "filename": (
                                header["fname"]
                                if header["fname"].startswith("/data/")
                                else None
                            ),
                        }
                        if "data" in header:
                            del header["data"]
                        for key in header:
                            if type(header[key]) is dict:
                                header[key] = str(header[key])
                        queue.put([meta_header, header])
                        in_scan = True
                        logger.debug("in_scan set to %s after new series", in_scan)

                    dtypes = {32: "uint32", 16: "uint16", 8: "uint8"}

                    data_header = {
                        "htype": "image",
                        "msg_number": next(self._msg_number),
                        "frame": header["frameIndex"],
                        "shape": (514, 514),  # header['shape'][::-1],
                        "type": dtypes[header["bitmode"]],
                        "compression": "none",
                    }

                    if len(d[0]) < 2 or len(d[1]) < 2:
                        end_header = {
                            "htype": "series_end",
                            "msg_number": next(self._msg_number),
                        }
                        logger.info("series end because no data available")
                        queue.put(
                            [
                                end_header,
                            ]
                        )
                        in_scan = False
                        continue
                    img = np.frombuffer(d[0][1].bytes, data_header["type"])
                    img = img.reshape((256, 512))
                    upper = np.flipud(img)

                    frame = np.zeros((514, 514), dtype=data_header["type"])

                    frame[0:256, 0:256] = upper[0:256, 0:256]
                    frame[0:256, 258:514] = upper[0:256, 256:]

                    img = np.frombuffer(d[1][1].bytes, data_header["type"])
                    lower = img.reshape((256, 512))

                    frame[258:, 0:256] = lower[0:256, 0:256]
                    frame[258:, 258:514] = lower[0:256, 256:]

                    flipped_frame = np.flipud(frame)
                    fb = np.ascontiguousarray(flipped_frame)

                    blob = fb.tobytes()

                    queue.put([data_header, blob])

            # clear single-modules responses
            for i in range(2):
                for k in list(hist[i].keys()):
                    print("oposite keys", list(hist[(i - 1) % 2].keys()))
                    if len(hist[(i - 1) % 2].keys()) > 0:
                        if k < min(hist[(i - 1) % 2].keys()):
                            print(
                                k,
                                " is smaller than any remaining in ",
                                hist[(i - 1) % 2].keys(),
                            )
                            print("Drop frame", k)
                            hist[i].pop(k)

            for k, v in prelast.items():
                last[k] = v
            prelast = {}
