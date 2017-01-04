require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'
ffi = require 'ffi'
torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
projectDir = paths.concat(os.getenv('HOME'),'pose-hg-train')

-- Process command line arguments, load helper functions
paths.dofile('opts.lua')
paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
if not Logger then paths.dofile('util/Logger.lua') end

-- Random number seed
if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
else torch.seed() end                           

-- Initialize dataset
if not dataset then
    local Dataset = paths.dofile(projectDir .. '/src/util/dataset/' .. opt.dataset .. '.lua')
    dataset = Dataset()
end

-- Global reference (may be updated in the task file below)
if not ref then
    ref = {}
    ref.nOutChannels = dataset.nJoints
    ref.inputDim = {3, opt.inputRes, opt.inputRes}
    ref.outputDim = {ref.nOutChannels, opt.outputRes, opt.outputRes}
end

-- Load up task specific variables / functions
paths.dofile('util/' .. opt.task .. '.lua')

-- Optimization function and hyperparameters
optfn = optim[opt.optMethod]
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = opt.LRdecay,
        momentum = opt.momentum,
        weightDecay = opt.weightDecay,
        alpha = opt.alpha,
        epsilon = opt.epsilon
    }
end

-- Print out input / output tensor sizes
if not ref.alreadyChecked then
    local function printDims(prefix,d)
        -- Helper for printing out tensor dimensions
        if type(d[1]) == "table" then
            print(prefix .. "table")
            for i = 1,#d do
                printDims("\t Entry " .. i .. " is a ", d[i])
            end
        else
            local s = ""
            if #d == 0 then s = "single value"
            elseif #d == 1 then s = string.format("vector of length: %d", d[1])
            else
                s = string.format("tensor with dimensions: %d", d[1])
                for i = 2,table.getn(d) do s = s .. string.format(" x %d", d[i]) end
            end
            print(prefix .. s)
        end
    end

    printDims("Input is a ", ref.inputDim)
    printDims("Output is a ", ref.outputDim)

    print("# of training images:", opt.idxRef.train:size(1))
    print("# of validation images:", opt.idxRef.valid:size(1))

    ref.alreadyChecked = true
end
