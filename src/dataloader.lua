--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and valid loader
   local loaders = {}

   for i, split in ipairs{'train', 'valid'} do
      loaders[split] = M.DataLoader(opt, split)
   end

   return loaders
end

function DataLoader:__init(opt, split)
   local function init()
         _G.opt = opt
         _G.split = split
         _G.alreadyChecked = true
         paths.dofile('ref.lua')
         paths.dofile('data.lua')
   end

   local function main(idx)
      torch.setnumthreads(1)
      if split == 'valid' then _G.isTesting = true end
      return opt[split .. 'Iters']*opt[split .. 'Batch']
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchsize = opt[split .. 'Batch']
   self.split = split
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchsize = self.__size, self.batchsize
   local perm = torch.randperm(size)

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchsize, size - idx + 1))
         threads:addjob(
            function(indices)
               local idx_ = nil
               if _G.isTesting then idx_ = idx end
               local inp,out = _G.loadData(_G.split, idx_, batchsize)
               collectgarbage()
               return {inp,out}
            end,

            function(_sample_)
               sample = _sample_
            end,

            indices
         )
         idx = idx + batchsize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
