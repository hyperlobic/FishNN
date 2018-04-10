module FishNN.Tests.Activations

open Xunit
open FsUnit.Xunit
open MathNet.Numerics.LinearAlgebra
open FishNN.Activations

let maxError = 0.00001

[<Fact>]
let ``sigmoid should calculate sigmoid activations for a matrix`` () =
    let m = matrix [[1.; 2.]; [3.; 4.]]
    
    let s = Sigmoid.sigmoid m

    s.[0,0] |> should (equalWithin maxError) 0.73105858
    s.[0,1] |> should (equalWithin maxError) 0.88079708
    s.[1,0] |> should (equalWithin maxError) 0.95257413
    s.[1,1] |> should (equalWithin maxError) 0.98201379

[<Fact>]    
let ``sigmoidDerivative should calculate derivatives of the sigmoid for a matrix`` () =
    let m = matrix [[1.; 2.]; [3.; 4.]]
    
    let s = Sigmoid.sigmoidDerivative m

    s.[0,0] |> should (equalWithin maxError) 0.19661193
    s.[0,1] |> should (equalWithin maxError) 0.10499359
    s.[1,0] |> should (equalWithin maxError) 0.04517666
    s.[1,1] |> should (equalWithin maxError) 0.01766271

[<Fact>]
let ``relu should calculate RELU activations for a matrix`` () =
    let m = matrix [[-1.; 2.]; [3.; -4.]]
    
    let s = Relu.relu m

    s.[0,0] |> should equal 0.
    s.[0,1] |> should equal 2.
    s.[1,0] |> should equal 3.
    s.[1,1] |> should equal 0.

[<Fact>]    
let ``reluDerivative should calculate derivatives of the RELU for a matrix`` () =
    let m = matrix [[-1.; 2.]; [3.; -44.]]
    
    let s = Relu.reluDerivative m

    s.[0,0] |> should equal 0.
    s.[0,1] |> should equal 1.
    s.[1,0] |> should equal 1.
    s.[1,1] |> should equal 0.