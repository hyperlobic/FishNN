module FishNN.Tests.Neural

open Xunit
open FsUnit.Xunit
open FishNN.Neural
open FishNN.Activations
open MathNet.Numerics.LinearAlgebra

[<Fact>]
let ``initNetwork should create valid parameter sizes`` () =
    let network = initNetwork [2; 3; 2] Relu.relu Relu.reluDerivative Relu.relu Relu.reluDerivative

    network.layers |> List.length |> should equal 2

    let layer1 = network.layers.[0]
    let layer2 = network.layers.[1]

    layer1.weights.RowCount |> should equal 3
    layer1.weights.ColumnCount |> should equal 2
    layer1.bias.Count |> should equal 3

    layer2.weights.RowCount |> should equal 2
    layer2.weights.ColumnCount |> should equal 3
    layer2.bias.Count |> should equal 2

[<Fact>]
let ``initLayer should create valid parameter sizes`` () =
    let layer = initLayer 4 2 Relu.relu Relu.reluDerivative

    layer.weights.RowCount |> should equal 4
    layer.weights.ColumnCount |> should equal 2
    layer.bias.Count |> should equal 4

[<Fact>]
let ``linearForwardProp should calculate linear portion of layer output`` () =
    let weights = matrix [[1.; 2.]; [3.; 4.]]
    let layer = {
        weights = weights
        bias = DenseVector.zero 2
        activation = Sigmoid.sigmoid
        activationDerivative = Sigmoid.sigmoidDerivative
    }

    let inputs = matrix [[3.; 10.]; [5.; -3.]]
    let linearOut = linearForwardProp inputs layer

    linearOut.ColumnCount |> should equal 1
    linearOut.RowCount |> should equal 2

    linearOut.[0,0] |> should equal 13.
    linearOut.[0,1] |> should equal 4.
    linearOut.[1,0] |> should equal 29.
    linearOut.[1,1] |> should equal 18.