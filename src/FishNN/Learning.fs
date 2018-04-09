module FishNN.Learning

open MathNet.Numerics.LinearAlgebra
open Neural

module Backpropagation =

    type LayerCache = {
        layer: Layer
        Z : Matrix<float>
        A_output : Matrix<float>
        A_input : Matrix<float>
    }

    type Gradients = {
        dW : Matrix<float>
        db : Vector<float>
    }

    let initCache inputs layer = {
        layer = layer
        Z = DenseMatrix.zero 0 0
        A_output = DenseMatrix.zero 0 0
        A_input = inputs
    }

    let layerForwardCached (inputs: Matrix<float>) layer =
        let z = linearForwardProp inputs layer
        let a = layer.activation z
        { layer = layer
          Z = z
          A_input = inputs
          A_output = a }

    let forwardPropCached inputs layers =
        let rec layerForward caches inputs layers =
            match layers with
            | [] -> caches
            | layer :: rest ->
                let cache = layerForwardCached inputs layer
                layerForward (cache :: caches) cache.A_output rest
        layerForward [] inputs layers

    let activationBackprop (dA : Matrix<float>) layerCache =
        let s = layerCache.layer.activationDerivative layerCache.Z
        dA .* s

    let linearBackprop (dZ : Matrix<float>) layerCache =
        let w = layerCache.layer.weights
        let m = layerCache.A_input.ColumnCount |> float
        let dW = (1. / m) * (dZ * layerCache.A_input.Transpose())
        let db = (1. / m) * (dZ |> Matrix.sumRows)
        let dAInput = w.Transpose() * dZ
        { dW = dW
          db = db }, dAInput

    let layerGradients (dZ : Matrix<float>) layerCache =
        let gradients' = layerCache |> activationBackprop dZ
        linearBackprop gradients' layerCache

    let networkGradients dCosts layerCaches =
        layerCaches
        |> List.fold(fun (grads, dZ) layerCache ->
            let g, dZNext = layerGradients dZ layerCache
            g :: grads, dZNext) ([], dCosts)
        |> fst

    let updateParameters network (learningRate : float) gradients =
        let newLayers =
            network.layers
            |> List.zip gradients
            |> List.map (fun (grads, layer) ->
                { layer with
                    weights = layer.weights - (learningRate * grads.dW)
                    bias = layer.bias - (learningRate * grads.db) })
        { network with layers = newLayers }

    let train learningRate costDerivative x y network =
        let caches = forwardPropCached x network.layers
        let output = List.head caches
        let dCosts = costDerivative output.A_output y

        let learnedNetwork =
            caches
            |> networkGradients dCosts
            |> updateParameters network learningRate

        learnedNetwork

