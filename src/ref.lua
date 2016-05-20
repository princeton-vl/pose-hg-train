-------------------------------------------------------------------------------
-- Load necessary libraries and files
-------------------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/Logger.lua')

torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
projectDir = paths.concat(os.getenv('HOME'),'pose-hg-train')

-------------------------------------------------------------------------------
-- Process command line options
-------------------------------------------------------------------------------

if not opt then

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

if opt.GPU == -1 then
    nnlib = nn
else
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    cutorch.setDevice(opt.GPU)
end

if opt.branch ~= 'none' or opt.continue then
    -- Continuing training from a prior experiment
    -- Figure out which new options have been set
    local setOpts = {}
    for i = 1,#arg do
        if arg[i]:sub(1,1) == '-' then table.insert(setOpts,arg[i]:sub(2,-1)) end
    end

    -- Where to load the previous options/model from
    if opt.branch ~= 'none' then opt.load = opt.expDir .. '/' .. opt.branch
    else opt.load = opt.expDir .. '/' .. opt.expID end

    -- Keep previous options, except those that were manually set
    local opt_ = opt
    opt = torch.load(opt_.load .. '/options.t7')
    opt.save = opt_.save
    opt.load = opt_.load
    opt.continue = opt_.continue
    for i = 1,#setOpts do opt[setOpts[i]] = opt_[setOpts[i]] end

    epoch = opt.lastEpoch + 1
    
    -- If there's a previous optimState, load that too
    if paths.filep(opt.load .. '/optimState.t7') then
        optimState = torch.load(opt.load .. '/optimState.t7')
        optimState.learningRate = opt.LR
    end

else epoch = 1 end
opt.epochNumber = epoch

-- Training hyperparameters
-- (Some of these aren't relevant for rmsprop which is the optimization we use)
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = opt.LRdecay,
        momentum = opt.momentum,
        dampening = 0.0,
        weightDecay = opt.weightDecay
    }
end

-- Optimization function
optfn = optim[opt.optMethod]

-- Random number seed
if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
else torch.seed() end                           

-- Save options to experiment directory
torch.save(opt.save .. '/options.t7', opt)

end

-------------------------------------------------------------------------------
-- Load in annotations
-------------------------------------------------------------------------------

annotLabels = {'train', 'valid'}
annot,ref = {},{}
for _,l in ipairs(annotLabels) do
    local a, namesFile
    if opt.dataset == 'mpii' and l == 'valid' and opt.finalPredictions == 1 then
        a = hdf5.open(opt.dataDir .. '/annot/test.h5')
        namesFile = io.open(opt.dataDir .. '/annot/test_images.txt')
    else
        a = hdf5.open(opt.dataDir .. '/annot/' .. l .. '.h5')
        namesFile = io.open(opt.dataDir .. '/annot/' .. l .. '_images.txt')
    end
    annot[l] = {}

    -- Read in annotation information
    local tags = {'part', 'center', 'scale', 'normalize', 'torsoangle', 'visible'}
    for _,tag in ipairs(tags) do annot[l][tag] = a:read(tag):all() end
    annot[l]['nsamples'] = annot[l]['part']:size()[1]

    -- Load in image file names (reading strings wasn't working from hdf5)
    annot[l]['images'] = {}
    local toIdxs = {}
    local idx = 1
    for line in namesFile:lines() do
        annot[l]['images'][idx] = line
        if not toIdxs[line] then toIdxs[line] = {} end
        table.insert(toIdxs[line], idx)
        idx = idx + 1
    end
    namesFile:close()

    -- This allows us to reference multiple people who are in the same image
    annot[l]['imageToIdxs'] = toIdxs

    -- Set up reference for training parameters
    ref[l] = {}
    ref[l].nsamples = annot[l]['nsamples']
    ref[l].iters = opt[l .. 'Iters']
    ref[l].batchsize = opt[l .. 'Batch']
    ref[l].log = Logger(paths.concat(opt.save, l .. '.log'), opt.continue)
end

ref.predict = {}
ref.predict.nsamples = annot.valid.nsamples
ref.predict.iters = annot.valid.nsamples
ref.predict.batchsize = 1

-- Default input is assumed to be an image and output is assumed to be a heatmap
-- This can change if an hdf5 file is used, or if opt.task specifies something different
nParts = annot['train']['part']:size(2)
dataDim = {3, opt.inputRes, opt.inputRes}
labelDim = {nParts, opt.outputRes, opt.outputRes}

-- Load up task specific variables/functions
-- (this allows a decent amount of flexibility in network input/output and training)
paths.dofile('util/' .. opt.task .. '.lua')

function applyFn(fn, t, t2)
    -- Helper function for applying an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end
