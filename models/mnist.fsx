#load "../.paket/load/net45/main.group.fsx"
#r "../src/FishNN/bin/Release/netcoreapp2.0/FishNN.dll"

open MathNet.Numerics.LinearAlgebra
open FishNN.Activations
open FishNN.Neural
open FishNN.Learning
open FishNN.Neural.Costs
open System
open System.IO

let dataFolder = Path.Combine(__SOURCE_DIRECTORY__, @"..\data\")

let read file =
    File.ReadAllLines file
    |> Array.map (fun s -> s.Split(','))
    |> Array.map (fun s -> s |> Array.map Double.Parse)
    |> Array.map (fun s -> Array.head s, Array.skip 1 s)
    |> Array.unzip

let initTargets (labels : float array) =
    let numSamples = labels.Length
    DenseMatrix.initColumns numSamples (fun c -> 
        let v = DenseVector.zero 10
        let index = labels.[c] |> int
        v.[index] <- 1.
        v)

let initInputs (data : float array array) =
    let numSamples = data.Length
    let x = DenseMatrix.initColumns numSamples (fun c -> DenseVector.ofArray(data.[c]))
    x / 255.

let trainingSet miniBatchSize =
    let labels, data = Path.Combine(dataFolder, "mnist_train.csv") |> read
    
    let labelBatches = 
        labels 
        |> Array.chunkBySize miniBatchSize 
        |> Array.map initTargets
    
    let dataBatches = 
        data 
        |> Array.chunkBySize miniBatchSize 
        |> Array.map initInputs
    
    dataBatches, labelBatches

let testingSet () =
    Path.Combine(dataFolder, "mnist_test.csv")
    |> read
    |> (fun (labels, data) -> (initInputs data, initTargets labels))


let miniBatchSize = 300
let learningRate = 0.1
let trainX, trainY = trainingSet miniBatchSize
let numBatches = trainX.Length
let testX, testY = testingSet ()
let numSamples = trainX |> Array.sumBy (fun x -> x.ColumnCount)
let numInputs = trainX.[0].RowCount
let numOutputs = trainY.[0].RowCount

printfn "inputs = %d, outputs = %d, batches = %d, samples = %d" numInputs numOutputs numBatches numSamples

let initialNetwork = initNetwork [numInputs; 50; 50; numOutputs] Relu.relu Relu.reluDerivative Sigmoid.sigmoid Sigmoid.sigmoidDerivative
let cost = CrossEntropyCosts.crossEntropy
let learn = Backpropagation.train learningRate CrossEntropyCosts.crossEntropyDerivative

let mutable network = initialNetwork

for epoch in 1..30 do

    printfn "Epoch %d" epoch

    for i in 0..numBatches - 1 do
        let x = trainX.[i]
        let y = trainY.[i]
        let result = learn x y network
        if i % 500 = 0 then
            let a = forwardProp x result
            printfn "cost = %s" ((cost a y).ToVectorString())
        network <- result

let mutable correctCount = 0

testX
|> Matrix.iteriCols (fun i col ->
    let y = predict col network
    let prediction = y.MaximumIndex()
    let actual = testY.Column(i).MaximumIndex()
    if prediction = actual then correctCount <- correctCount + 1)

let numTestSamples = testX.ColumnCount

let accuracy = (float correctCount) / (float numTestSamples)
printfn "Correctly predicted %d / %d - accuracy = %f" correctCount numTestSamples accuracy
