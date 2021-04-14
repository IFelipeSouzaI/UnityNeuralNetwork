using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork
{
    private int input_nodes;
    private int hidden_layers;
    private int hidden_nodes;
    private int output_nodes;

    private FMatrix[] HIDDEN;
    private FMatrix[] WEIGHTS;
    private FMatrix[] BIAS;

    private FMatrix weights_ih;
    private FMatrix weights_ho;
    private FMatrix bias_h;
    private FMatrix bias_o;
    private float learning_rate = 0.1f;

    public NeuralNetwork(int inputsNodesAmount, int hiddenNodesAmount, int outputNodesAmount, int hiddenLayersAmount = 1){
        input_nodes = inputsNodesAmount;
        hidden_nodes = hiddenNodesAmount;
        output_nodes = outputNodesAmount;
        hidden_layers = hiddenLayersAmount;

        WEIGHTS = new FMatrix[hidden_layers+1];
        WEIGHTS[0] = new FMatrix(hidden_nodes, input_nodes);
        for(int i = 1; i < hidden_layers; i++){
            WEIGHTS[i] = new FMatrix(hidden_nodes, hidden_nodes);    
        }
        WEIGHTS[hidden_layers] = new FMatrix(output_nodes, hidden_nodes);
        
        BIAS = new FMatrix[hidden_layers+1];
        for(int i = 0; i < hidden_layers; i++){
            BIAS[i] = new FMatrix(hidden_nodes, 1);
        }
        BIAS[hidden_layers] = new FMatrix(output_nodes, 1);

        for(int i = 0; i < hidden_layers; i++){
            WEIGHTS[i].Randomize();
            BIAS[i].Randomize();
        }

        /*
        weights_ih = new FMatrix(hidden_nodes, input_nodes);
        weights_ho = new FMatrix(output_nodes, hidden_nodes);
        weights_ih.Randomize();
        weights_ho.Randomize();

        bias_h = new FMatrix(hidden_nodes, 1);
        bias_o = new FMatrix(output_nodes, 1);
        bias_h.Randomize();
        bias_o.Randomize();*/

    }
    public FMatrix FeedFoward(float[] inputsArray){
        FMatrix inputs = FMatrix.Array2Matrix(inputsArray);
        
        HIDDEN = new FMatrix[hidden_layers];
        HIDDEN[0] = FMatrix.newByProduct(WEIGHTS[0], inputs, "FeedFoward Hidden[0]");
        HIDDEN[0].PlusM(BIAS[0]);
        HIDDEN[0].SigmoidMap();
        for(int i = 1; i < hidden_layers; i++){
            HIDDEN[i] = FMatrix.newByProduct(WEIGHTS[i], HIDDEN[i-1], "FeedFoward For");
            HIDDEN[i].PlusM(BIAS[i]);
            HIDDEN[i].SigmoidMap();
        }
        FMatrix outputs = FMatrix.newByProduct(WEIGHTS[hidden_layers], HIDDEN[hidden_layers-1], "FeedForward Output");
        outputs.PlusM(BIAS[hidden_layers-1]);
        outputs.SigmoidMap();
        return outputs;
    }
/*
    public FMatrix FeedFoward(float[] i){
        FMatrix inputs = FMatrix.Array2Matrix(i);

        FMatrix hidden = FMatrix.newByProduct(weights_ih, inputs);
        hidden.PlusM(bias_h);
        hidden.SigmoidMap();

        FMatrix outputs = FMatrix.newByProduct(weights_ho, hidden);
        outputs.PlusM(bias_o);
        outputs.SigmoidMap();
        
        return outputs;
    }
*/
    public void Train(float[] inputsArray, float[] targetsArray){

        FMatrix inputs = FMatrix.Array2Matrix(inputsArray);
        FMatrix outputs = FeedFoward(inputsArray);
        FMatrix targets = FMatrix.Array2Matrix(targetsArray);
        
        FMatrix output_errors = FMatrix.newByMinus(targets, outputs);
        FMatrix outputGradient = FMatrix.newGradientM(outputs); // Output -> Gradiente
        outputGradient.ProductM(output_errors); // -
        outputGradient.ProductE(learning_rate); // -->
        FMatrix hidden_transpose = FMatrix.newByTranspose(HIDDEN[hidden_layers-1]); 
        FMatrix weights_delta = FMatrix.newByProduct(outputGradient, hidden_transpose, "outputGradient");
        WEIGHTS[hidden_layers].PlusM(weights_delta);
        BIAS[hidden_layers].PlusM(outputGradient);

        for(int i = hidden_layers-1; i > 0; i--){
            FMatrix weights_transpose = FMatrix.newByTranspose(WEIGHTS[hidden_layers]);
            FMatrix h_errors = FMatrix.newByProduct(weights_transpose, output_errors, "hidden_erros do for");
            FMatrix hiddenGradient = FMatrix.newGradientM(HIDDEN[i]); // Hidden -> Gradient
            hiddenGradient.ProductM(h_errors); // -
            hiddenGradient.ProductE(learning_rate); // -->
            FMatrix h_transpose = FMatrix.newByTranspose(HIDDEN[i-1]);
            FMatrix weights_deltas = FMatrix.newByProduct(hiddenGradient, h_transpose, "weights_delta do for");
            WEIGHTS[i].PlusM(weights_deltas);
            BIAS[i].PlusM(hiddenGradient);
        }

        FMatrix weights_ho_transpose = FMatrix.newByTranspose(WEIGHTS[hidden_layers]);
        //Debug.Log(weights_ho_transpose.rows + " " + weights_ho_transpose.columns + " <--> " + output_errors.rows + " " + output_errors.columns);
        FMatrix hidden_errors = FMatrix.newByProduct(weights_ho_transpose, output_errors, "hidden_erros");
        FMatrix inputGradient = FMatrix.newGradientM(HIDDEN[0]); // Inputs -> Gradiente
        inputGradient.ProductM(hidden_errors); // -
        inputGradient.ProductE(learning_rate); // -->
        FMatrix inputs_transpose = FMatrix.newByTranspose(inputs);
        FMatrix weights_ih_deltas = FMatrix.newByProduct(inputGradient, inputs_transpose, "weights_hi_delta");
        WEIGHTS[0].PlusM(weights_ih_deltas);
        BIAS[0].PlusM(inputGradient);
        /*
        Debug.Log("Output: " + outputs.Print());
        Debug.Log("Targets: " + targets.Print());
        Debug.Log("Error: " + output_errors.Print());   
        */
    }
}