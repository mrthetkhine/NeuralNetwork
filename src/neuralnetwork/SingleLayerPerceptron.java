/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import java.util.ArrayList;

/**
 *
 * @author Mr T.Khine
 */
public class SingleLayerPerceptron {
    
    float[] weight;
    float[] bias;
    float[] output;
    
    float[] currentInput;
    float[] currentOutput;
    
    float[] errors;
    float learningRate =0.2f;
    ArrayList<Pattern> data;
    
    int noOfInput;
    int noOfOutput;
    public SingleLayerPerceptron(int noOfInput,int noOfOutput,ArrayList<Pattern> data)
    {
        this.noOfInput = noOfInput;
        this.noOfOutput = noOfOutput;
        this.weight = new float[this.noOfInput];
        this.bias= new float[this.noOfOutput];
        this.output = new float[this.noOfOutput];
        this.errors = new float[this.noOfOutput];
        this.data = data;
        
    }
    public void initPerceptron()
    {
        for (int i = 0; i < this.noOfInput; i++) 
        {
            this.weight[i] =(float) Math.random()  -0.5f;
        }
        for (int i = 0; i < this.noOfOutput; i++) {
            this.bias[i] = (float)(Math.random() );
        }
    }
    public void run(int iteration)
    {
        this.initPerceptron();
        for (int i = 0; i < iteration; i++) 
        {
            trainEpoch();
            
        }
    }
    float totalError = 0;
    private void trainEpoch() {
         totalError = 0;
        //load a random pattern from the training example
        //Shuffle random index for data
        ArrayList<Integer> randomIndex = new ArrayList<Integer>();
        for (int j = 0; j < data.size(); j++)
        {
            randomIndex.add(j);
        }
        //go through all trainning example
        while(randomIndex.size() !=0)
        {
            //pick a random index;
            int random = (int)( Math.random()* randomIndex.size());
            int trainingIndex = randomIndex.get(random);
            randomIndex.remove(random);
            
            this.currentInput = this.data.get(trainingIndex).input;
            this.currentOutput = this.data.get(trainingIndex).output;
            
            //Compute output
            for (int out = 0; out < this.noOfOutput; out++) {
                //clear output for previous step
                this.output[out] = 0;
                
                //output is sum of input and weight + bias
                for (int input = 0; input < this.noOfInput; input++) {
                    this.output[out]+= this.currentInput[input]* this.weight[input];
                }
                for (int bias = 0; bias < this.noOfOutput; bias++) {
                    this.output[out] += this.bias[bias];
                }
                //then go through activation fucntion
                //System.out.println("Output "+ this.output[out]+ " Real Output "+ this.currentOutput[out]);
                this.output[out] = this.activation(this.output[out]);
                this.errors[out] = this.currentOutput[out] - this.output[out] ;
         
            }
            //compute total eror to update weight
            //totalError = 0;
            for (int i = 0; i < this.noOfOutput; i++) {
                totalError += 0.5 * Math.pow(this.errors[i],2);
            }
            
            //System.out.println("Error "+ this.errors[0]);
            //update weight
            for (int i = 0; i < this.noOfInput; i++) {
                this.weight[i] = this.weight[i] +this.currentInput[i] * this.errors[0] * learningRate;
            }
            //update bias
            for (int i = 0; i < this.noOfOutput; i++) {
                this.bias[i] = this.bias[i] * this.currentInput[i] + this.errors[0] * this.learningRate;
            }
        }
        totalError = totalError / this.data.size();
        //System.out.println("Total error "+ totalError);
    }
    public int activation(float value)
    {
        return value>2? 1 :0;
    }
    public float feedForward(float[] inputs)
    {
        for (int out = 0; out < this.noOfOutput; out++) 
        {
            //clear output for previous step
            this.output[out] = 0;

            //output is sum of input and weight + bias
            for (int input = 0; input < this.noOfInput; input++) 
            {
                this.output[out]+= inputs[input]* this.weight[input];
            }
            for (int bias = 0; bias < this.noOfOutput; bias++) {
                this.output[out] += this.bias[bias];
            }
            //then go through activation fucntion
            //System.out.println("Output "+ this.output[out]+ " Real Output "+ this.currentOutput[out]);
            this.output[out] = this.activation(this.output[out]);
    
        }
        return this.output[0];
    }
    public void showWeightAndBias()
    {
        
        for (int i = 0; i < this.noOfInput; i++) {
            System.out.println("Weight "+i +" "+this.weight[i]);
        }
        for (int i = 0; i < this.noOfOutput; i++) {
            System.out.println("Bias "+ i + " "+this.bias[i]);
        }
    }
    public static void main(String[] args) {
        
        Pattern pattern = new Pattern(new float[]{0,0},new float[]{0});
        Pattern pattern2 = new Pattern(new float[]{0,1},new float[]{1});
        Pattern pattern3 = new Pattern(new float[]{1,0},new float[]{1});
        Pattern pattern4 = new Pattern(new float[]{1,1},new float[]{1});
        
        ArrayList<Pattern> data = new ArrayList<Pattern>();
        data.add(pattern);
        data.add(pattern2);
        data.add(pattern3);
        data.add(pattern4);
        
        SingleLayerPerceptron neural = new SingleLayerPerceptron(2,1,data);
        neural.run(130);
        
        neural.showWeightAndBias();
        
        System.out.println("Nerual test 0,0 "+ neural.feedForward( new float[]{0,0}));
        System.out.println("Nerual test 0,1 "+ neural.feedForward( new float[]{0,1}));
        System.out.println("Nerual test 1,0 "+ neural.feedForward( new float[]{1,0}));
        System.out.println("Nerual test 1,1 "+ neural.feedForward( new float[]{1,1}));
    }
    
}
