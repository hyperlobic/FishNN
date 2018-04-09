module FishNN.Neural

open MathNet.Numerics.LinearAlgebra
open Activations

module Costs =

    module CrossEntropyCosts =

        let crossEntropy (AL : Matrix<float>) (Y : Matrix<float>) =
            let m = Y.ColumnCount |> float
            (- (Y .* AL.PointwiseLog())) - ((1. - Y) .* ((1. - AL).PointwiseLog()))
            |> Matrix.sumRows
            |> (*) (1. / m)

        let crossEntropyDerivative (AL : Matrix<float>) (Y : Matrix<float>) = - (( Y ./ AL ) - ( (1. - Y) ./ (1. - AL)))

type Weights = Matrix<float>
type Bias = Vector<float>

type Layer = {
    weights: Weights
    bias : Bias
    activation: Activation
    activationDerivative : Activation
}

type Network = { layers : Layer list }

let initLayer size inputSize activation derivative = 
    { weights = (DenseMatrix.randomStandard size inputSize) * 0.01
      bias = DenseVector.zero size
      activation = activation
      activationDerivative = derivative }

let initNetwork layerSizes hiddenActivation hiddenDerivative outputActivation outputDerivative =
    let sizes = List.pairwise layerSizes
    let hidden = List.take (List.length sizes - 1) sizes
    let lastHiddenSize, outputSize = List.last sizes

    let hiddenLayers =
        hidden
        |> List.map (fun (prev, curr) -> initLayer curr prev hiddenActivation hiddenDerivative)

    let outputLayer = initLayer outputSize lastHiddenSize outputActivation outputDerivative

    { layers = hiddenLayers @ [ outputLayer ] }

let linearForwardProp (inputs : Matrix<float>) layer  =
    let z = layer.weights * inputs
    Matrix.mapColsInPlace (fun _ col -> col + layer.bias) z
    z

let layerForward (inputs : Matrix<float>) layer =
    layer
    |> linearForwardProp inputs
    |> layer.activation

let forwardProp (inputs : Matrix<float>) network =
    network.layers
    |> List.fold layerForward inputs

let predict (inputs : Vector<float>) network =
    let x = inputs.ToColumnMatrix()
    forwardProp x network
    |> fun x -> x.Column(0)









