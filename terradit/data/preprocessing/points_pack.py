"""Packed point-supervision format for the sigma model.

The original sigma supervision is ~1.95M dense 256x256 int tag tensors (.pt) plus
per-tile taglist JSONs (~510 GB unzipped, millions of files). Each tile has only a
handful of distinct values in large contiguous regions, so we pack every tile as
RLE + zlib into a single indexed file (~hundreds of MB) that stays compressed on
disk and decodes per tile in microseconds.

Per tile we store exactly what the dataset needs:
  - the dense 256x256 tag tensor (values = per-tile taglist indices, -1 = no OSM)
  - the index_to_taglist mapping (per-tile taglist index -> list of global tag ids)

`PointPackReader.get(google_location)` mirrors `LegacyPixelTensorSource.get`:
returns (pixel_tensor [256,256] long, index_to_taglist dict | None). A tile absent
from the pack yields an all -1 tensor (no coverage), matching the original.

File layout:
    [record]*            record = u32 pt_len | pt_blob | u32 tags_len | tags_blob
    [index json]         {google_location: byte_offset_of_record}
    u64 index_offset
    8-byte magic 'GDPTPCK1'
"""
import os
import io
import json
import zlib
import struct

import numpy as np
import torch

MAGIC = b"GDPTPCK1"
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
TILE_HW = 256


# --------------------------- RLE codec --------------------------- #
def rle_encode(tensor2d):
    """Encode a 2D int tag tensor -> bytes (H, W, runs of (value, length))."""
    a = np.asarray(tensor2d, dtype=np.int16).ravel()
    H, W = (tensor2d.shape if hasattr(tensor2d, "shape") else (TILE_HW, TILE_HW))
    if a.size == 0:
        starts = np.array([], dtype=np.int64)
        vals = np.array([], dtype=np.int16)
        lens = np.array([], dtype=np.int32)
    else:
        change = np.flatnonzero(np.diff(a)) + 1
        starts = np.concatenate(([0], change))
        vals = a[starts].astype(np.int16)
        lens = np.diff(np.concatenate((starts, [a.size]))).astype(np.int32)
    header = struct.pack("<HHI", H, W, len(vals))
    payload = header + vals.tobytes() + lens.tobytes()
    return zlib.compress(payload, 6)


def rle_decode(blob):
    """Decode bytes produced by rle_encode -> long tensor [H, W]."""
    payload = zlib.decompress(blob)
    H, W, n = struct.unpack_from("<HHI", payload, 0)
    off = 8
    vals = np.frombuffer(payload, dtype=np.int16, count=n, offset=off)
    off += n * 2
    lens = np.frombuffer(payload, dtype=np.int32, count=n, offset=off)
    if n == 0:
        flat = np.full(H * W, -1, dtype=np.int64)
    else:
        flat = np.repeat(vals.astype(np.int64), lens.astype(np.int64))
    return torch.from_numpy(flat.reshape(H, W).copy()).long()


def _encode_tags(index_to_taglist):
    obj = index_to_taglist if index_to_taglist is not None else {}
    return zlib.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"), 6)


def _decode_tags(blob):
    obj = json.loads(zlib.decompress(blob).decode("utf-8"))
    return obj if obj else None


# --------------------------- Writer --------------------------- #
class PointPackWriter:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "wb")
        self.index = {}

    def add(self, google_location, pixel_tensor, index_to_taglist):
        offset = self.f.tell()
        pt_blob = rle_encode(pixel_tensor)
        tags_blob = _encode_tags(index_to_taglist)
        self.f.write(_U32.pack(len(pt_blob)))
        self.f.write(pt_blob)
        self.f.write(_U32.pack(len(tags_blob)))
        self.f.write(tags_blob)
        self.index[google_location] = offset

    def close(self):
        index_offset = self.f.tell()
        self.f.write(json.dumps(self.index, separators=(",", ":")).encode("utf-8"))
        self.f.write(_U64.pack(index_offset))
        self.f.write(MAGIC)
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------- Reader --------------------------- #
class PointPackReader:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            f.seek(-len(MAGIC), io.SEEK_END)
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{path} is not a valid points.pack (bad magic)")
            f.seek(-(len(MAGIC) + _U64.size), io.SEEK_END)
            index_offset = _U64.unpack(f.read(_U64.size))[0]
            f.seek(index_offset)
            blob = f.read()  # up to the footer pointer
            blob = blob[: len(blob) - (_U64.size + len(MAGIC))]
            self.index = json.loads(blob.decode("utf-8"))
        self._fh = {}  # per-process file handle (dataloader workers fork)

    def _handle(self):
        pid = os.getpid()
        fh = self._fh.get(pid)
        if fh is None:
            fh = open(self.path, "rb")
            self._fh[pid] = fh
        return fh

    def __contains__(self, google_location):
        return google_location in self.index

    def get(self, google_location):
        if google_location not in self.index:
            return -1 * torch.ones((TILE_HW, TILE_HW), dtype=torch.long), None
        fh = self._handle()
        fh.seek(self.index[google_location])
        pt_len = _U32.unpack(fh.read(_U32.size))[0]
        pt_blob = fh.read(pt_len)
        tags_len = _U32.unpack(fh.read(_U32.size))[0]
        tags_blob = fh.read(tags_len)
        return rle_decode(pt_blob), _decode_tags(tags_blob)
