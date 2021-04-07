using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork
{
    private int input_nodes;
    private int hidden_layers;
    private int hidden_nodes;
    private int output_nodes;

    private FMatrix weights_ih;
    private FMatrix weights_ho;
    private FMatrix bias_h;
    private FMatrix bias_o;
    private float learning_rate = 0.1f;

    public NeuralNetwork(int i, int h, int o){
        input_nodes = i;
        hidden_nodes = h;
        output_nodes = o;

        weights_ih = new FMatrix(hidden_nodes, input_nodes);
        weights_ho = new FMatrix(output_nodes, hidden_nodes);
        weights_ih.Randomize();
        weights_ho.Randomize();

        bias_h = new FMatrix(hidden_nodes, 1);
        bias_o = new FMatrix(output_nodes, 1);
        bias_h.Randomize();
        bias_o.Randomize();

    }

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

    public void Train(float[] i, float[] t){

        FMatrix inputs = FMatrix.Array2Matrix(i); // Transforma os inputs em uma matriz
        FMatrix hidden = FMatrix.newByProduct(weights_ih, inputs, "hidden"); // Creating the hidden Layer
        hidden.PlusM(bias_h); // Adciona bias
        
        hidden.SigmoidMap(); // Chama a sigmoid pra deixar entre 0 e 1

        FMatrix outputs = FMatrix.newByProduct(weights_ho, hidden, "output"); // Outputs são = ao produto da ultima hidden layer pela soma de pesos do ultimo par de layers 
        outputs.PlusM(bias_o); // Adiciona o Bias
        outputs.SigmoidMap(); // Coloca entre 0 e 1

        FMatrix targets = FMatrix.Array2Matrix(t);

        FMatrix output_errors = FMatrix.newByMinus(targets, outputs);
        
        outputs.SigmoidMap(true); // Gradiente
        outputs.ProductM(output_errors);
        outputs.ProductE(learning_rate);

        FMatrix hidden_transpose = FMatrix.newByTranspose(hidden); 
        FMatrix weights_ho_delta = FMatrix.newByProduct(outputs, hidden_transpose, "weights_ho_delta");
        
        weights_ho.PlusM(weights_ho_delta);
        
        bias_o.PlusM(outputs);

        FMatrix weights_ho_transpose = FMatrix.newByTranspose(weights_ho);
        FMatrix hidden_errors = FMatrix.newByProduct(weights_ho_transpose, output_errors, "hidden_erros");

        hidden.SigmoidMap(true);
        hidden.ProductM(hidden_errors);
        hidden.ProductE(learning_rate);

        FMatrix inputs_transpose = FMatrix.newByTranspose(inputs);
        FMatrix weights_ih_deltas = FMatrix.newByProduct(hidden, inputs_transpose, "weights_hi_delta");
        
        weights_ih.PlusM(weights_ih_deltas);
        
        bias_h.PlusM(hidden);
        /*
        Debug.Log("Output: " + outputs.Print());
        Debug.Log("Targets: " + targets.Print());
        Debug.Log("Error: " + output_errors.Print());   
        */
    }
}