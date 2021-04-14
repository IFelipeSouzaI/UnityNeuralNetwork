using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class main : MonoBehaviour
{

    private float[] i0 = new float[] {0,1};
    private float[] i1 = new float[] {1,0};
    private float[] i2 = new float[] {1,1};
    private float[] i3 = new float[] {0,0};
    private float[] target0 = new float[] {1};
    private float[] target1 = new float[] {1};
    private float[] target2 = new float[] {0};
    private float[] target3 = new float[] {0};

    void Start()
    {
        int hidden_layers = 2;
        NeuralNetwork nn = new NeuralNetwork(2,5,1,hidden_layers);

        for(int i = 0; i < 50000; i++){
            int r = Random.Range(0,10)%4;
            if(r == 0){
                nn.Train(i0,target0);
            }else if(r == 1){
                nn.Train(i1, target1);
            }else if(r == 2){
                nn.Train(i2, target2);
            }else if(r == 3){
                nn.Train(i3, target3);
            }
        }

        Debug.Log(nn.FeedFoward(i0).Print());
        Debug.Log(nn.FeedFoward(i1).Print());
        Debug.Log(nn.FeedFoward(i2).Print());
        Debug.Log(nn.FeedFoward(i3).Print());
    }
}


