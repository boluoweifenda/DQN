class Options:

    Name = 'temp'
    timeFormat = '%Y-%m-%d %X'
    dataType = 'float32'

    device = '/gpu:2'

    pathLog = "../log/" + Name + ".txt"
    pathModel = "../model/" + Name + ".tfmodel"
    pathData = "../data/"
    pathRom = "../rom/breakout.bin"    # breakout space_invaders seaquest pong ms_pacman montezuma_revenge

    loadData = False
    loadModel = False
    saveData = False
    saveModel = True
    preTraining = 1000000

    NNType = 'CNN'
    RLType = 'Value' #'Policy Gradient'
    dataFormat = 'NCHW'
    discountFactor = .99
    lifeReward = 0



    batchSize = 32
    historyLength = 4
    height = 84
    width = 84
    memorySize = 1000000

    optimizer = "Adam"
    learningRate = 0.0001
    decay = 0.9
    momentum = 0.0
    epsilon = 1e-8
    belta1 = 0.95
    belta2 = 0.999

    frameStartLearn = 500
    frameFinalExp = 1000
    frameMax = 50000000
    freqSaveData = 10000000
    freqSaveModel = 2000000
    freqTrain = 4
    freqUpdate = 10000
    freqTest = 10000000
    frameTest = 200000

    loadRatio = -1

    expInit = 1.1
    expFinal = 0.1
    expTest = 0.05

    environment = "ALE" #'GYM'
    show = False
    delay = 1
    frameSkip = 4
    colorAverageing = False
    phosphorBlendRatio = 100  # 0-min 100-max 77-default
    maxNumFramesPerEpisode = 0

    maxminRatio = .7 # 1-max 0-min
    noopMax = 0

    Notes = "Q learning" \
            "just a test"


    # to be defined shared parameters
    randomSeed = 20161013
    n_actions = None

