require 'paths'
paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('data.lua')    -- Set up data processing
paths.dofile('model.lua')   -- Read in network model
paths.dofile('train.lua')   -- Load up training/testing functions

isFinished = false -- Finish early if validation accuracy plateaus, can be adjusted with opt.threshold

-- Main training loop
for i=1,opt.nEpochs do
    train()
    valid()
    collectgarbage()
    epoch = epoch + 1
    if isFinished then break end
end

-- Update options/reference for last epoch
opt.lastEpoch = epoch - 1
torch.save(opt.save .. '/options.t7', opt)

-- Generate final predictions on validation set
if opt.finalPredictions == 1 then predict() end

-- Save model
model:clearState()
torch.save(paths.concat(opt.save,'final_model.t7'), model)
torch.save(paths.concat(opt.save,'optimState.t7'), optimState)
