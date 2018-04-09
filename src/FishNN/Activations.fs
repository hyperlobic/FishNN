module FishNN.Activations

open System
open MathNet.Numerics.LinearAlgebra

type Activation = Matrix<float> -> Matrix<float>

module Sigmoid =

    let sigmoid (z : Matrix<float>) = 1. / (1. + (Matrix.Exp -z))

    let sigmoidDerivative (z : Matrix<float>) = 
        let s = 1. / (1. + (Matrix.Exp -z))
        s .* (1. - s)


module Relu = 
    
    let relu (z : Matrix<float>) = z |> Matrix.map(fun x -> Math.Max(0., x))
   
    let reluDerivative (z : Matrix<float>) = z |> Matrix.map(fun x -> if x < 0. then 0. else 1.)