#!/usr/bin/env python
'''
tred.blocking provides functions for N-dimensional "blocks".
'''

from tred.util import to_tensor
import torch
from .types import IntTensor, Tensor, Shape, Size, index_dtype

class Block:
    '''
    A batch of rectangular volumes at locations in a discrete integer space.

    Each volume is restricted to sharing a common (unbatched) `shape`.
    '''
    def __init__(self, location:IntTensor, shape: Shape|None=None, data:Tensor|None=None):
        '''
        Create a Block.

        The `location` of the volumes may be 1D (unbatched) or 2D (batched)
        N-vector.  If unbatched, a single-batch version is stored.  If not
        proved as a tensor, it is coerced to an IntTensor on the default device.

        The `shape` gives the common volume shape.

        The `data` provides a tensor of values defined on the volumes.  It will
        be stored in batched form consistent with that of `location`.

        The `shape` argument is only considered if `data` is not provided.
        '''
        if not isinstance(location, Tensor):
            location = torch.tensor(location, dtype=index_dtype)

        if len(location.shape) == 1:
            location = location.unsqueeze(0)
        if len(location.shape) != 2:
            raise ValueError(f'Block: unsupported shape for volume locations: {location.shape}')
        self.location = location

        if data is None:
            self.set_shape(shape)
        else:
            self.set_data(data)

    def size(self):
        '''
        Return torch.Size like a tensor.size() does. This includes batched dimension.
        '''
        return Size([self.nbatches] + self.shape.tolist())

    @property
    def vdim(self):
        '''
        Return the number of spacial/vector dimensions, excluding batch dimension
        '''
        return self.location.shape[1]

    @property
    def nbatches(self):
        '''
        Return the number of batches.
        '''
        return self.location.shape[0]

    def set_shape(self, shape:Shape):
        '''
        Set the spacial shape of the volumes.  This will drop any data.
        '''
        if hasattr(self, "data"):
            delattr(self, "data")

        if not isinstance(shape, Tensor):
            shape = self.to_tensor(shape, dtype=index_dtype)

        if len(shape.shape) != 1:
            raise ValueError(f'Block: volume shape must not be batched: {shape}')
        vdim = self.vdim
        if len(shape) != vdim:
            raise ValueError(f'Block: volume shape has wrong dimensions: {len(shape)} expected {vdim}')
        self.shape = shape

        
    def set_data(self, data:Tensor):
        '''
        Set data for block.
        '''
        if not isinstance(data, Tensor):
            raise TypeError('Block: data must be a tensor')

        vdim = self.vdim

        if len(data.shape) == vdim:
            data = data.unsqueeze(0)
        if len(data.shape) != vdim+1:
            raise ValueError(f'Block: illegal data shape: {data.shape} for volume {vdim}-dimension volumes')
        nbatch = self.nbatches
        if data.shape[0] != nbatch:
            raise ValueError(f'Block: batch mismatch: got {data.shape[0]} want {nbatch}')

        self.set_shape(data.shape[1:])
        self.data = data

    def to_tensor(self, thing, dtype=index_dtype):
        '''
        Return thing as tensor on same device as location tensor.
        '''
        return to_tensor(thing, device=self.location.device, dtype=dtype)


def batchify(block: Block) -> Block:
    '''
    Return block which is assured to be batched.
    '''
    nl = len(block.location.shape)
    nd = len(block.data.shape)

    if nl == 1:             # nominally unbatched
        vdim = block.location.shape[0]
        if vdim != nd:
            raise ValueError(f'unbatched block dimension mismatch {nd} != {vdim}')
        return Block(location=torch.unsqueeze(block.location,0),
                     data=torch.unsqueeze(block.data,0))

    bd = block.data.shape[0]
    bl = block.location.shape[0]
    if bd != bl:
        raise ValueError(f'batch size mismatch {bd} != {bl}')


    vdim = block.location.shape[1]
    if vdim+1 != nd:
        raise ValueError(f'batched block dimension mismatch {nd} != {vdim+1}')

    return block
