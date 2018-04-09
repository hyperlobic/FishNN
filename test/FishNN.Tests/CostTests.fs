module FishNN.Tests.Costs

open Xunit
open FsUnit.Xunit
open MathNet.Numerics.LinearAlgebra
open FishNN.Neural.Costs.CrossEntropyCosts

let maxError = 0.00001

[<Fact>]
let ``crossEntropy should calculate cross entropy costs`` () =
    let a = matrix [[0.2; 0.8; 0.3]; [0.4; 0.2; 0.1]]
    let y = matrix [[0.; 1.; 1.]; [1.; 0.; 0.;]]

    let costs = crossEntropy a y

    costs.[0] |> should (equalWithin maxError) 0.55008664
    costs.[1] |> should (equalWithin maxError) 0.4149316

[<Fact>]
let ``crossEntropyDerivative should calculate derivative of the cross entropy cost`` () =
    let a = matrix [[0.2; 0.8; 0.3]; [0.4; 0.2; 0.1]]
    let y = matrix [[0.; 1.; 1.]; [1.; 0.; 0.;]]

    let dCosts = crossEntropyDerivative a y

    dCosts.[0,0] |> should (equalWithin maxError)  1.25
    dCosts.[0,1] |> should (equalWithin maxError)  -1.25
    dCosts.[0,2] |> should (equalWithin maxError)  -3.33333333
    dCosts.[1,0] |> should (equalWithin maxError)  -2.5 
    dCosts.[1,1] |> should (equalWithin maxError)  1.25
    dCosts.[1,2] |> should (equalWithin maxError)  1.11111111
        