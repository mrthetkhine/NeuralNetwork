package neuralnetwork;

import java.util.*;


public class FeedForwardNeuralNetwork {
    int noOfInputNeuron;
    int noOfOutputNeuron;
    int noOfHiddenNeuron;
    float learningRate;


    float[] inputNeuron; // store actual output from inputNeuron node
    float[] hiddenNeuron;
    float[] outputNeuron;
    
    float[][] weightFromInputToHidden;
    float[][] weightFromHiddenToOutput;

    float[] biasForHiddenNode;
    float[] biasForOutputNode;
    float[] outputOfOutputNeuron; //Output produced by output neuron
    
    float[] errorOfOutputLayer; //Used for back propagation
    float[] errorOfHiddenLayer;

    Vector inputPatterns = new Vector();
    Vector targetValues =  new Vector();
    float[] actualTargetValues ; //For Trainning

    int currentInputPattern = 0; //Index to current trainning pattern

    ///error of the network, to calculat mean square error
    float[] meanSquareError ;
    float meanError ;
    float accuracy;

    ArrayList<Integer> indexOfPattern = new ArrayList<Integer>();
    public FeedForwardNeuralNetwork(int inputNode,int hiddenNode,int outputNode)
    {
        //Allocate data
        this.noOfInputNeuron = inputNode;
        this.noOfHiddenNeuron = hiddenNode;
        this.noOfOutputNeuron = outputNode;

        this.inputNeuron = new float[ this.noOfInputNeuron ];
        this.hiddenNeuron = new float[ this.noOfHiddenNeuron];
        this.outputNeuron = new float[ this.noOfOutputNeuron ] ;

        //Intialize weight 
        this.weightFromInputToHidden = new float[ this.noOfInputNeuron ][this.noOfHiddenNeuron];
        this.weightFromHiddenToOutput = new float[ this.noOfHiddenNeuron ][this.noOfOutputNeuron];

        this.biasForHiddenNode = new float[ this.noOfHiddenNeuron];
        this.biasForOutputNode = new float[ this.noOfOutputNeuron];

        this.errorOfHiddenLayer = new float[ this.noOfHiddenNeuron ];
        this.errorOfOutputLayer = new float[ this.noOfOutputNeuron ];
        this.learningRate = 0.001f;

        this.meanSquareError = new float[this.noOfOutputNeuron];
        initializeWeightToNetwork();
    


    }
    public void setLearningRate(float learnRate)
    {
        this.learningRate = learnRate;
    }
    float getRandom()
    {
        double d = Math.random() -0.5F; //From 0.5 to -0.5;
        return (float)(d);
    }
    void initializeWeightToNetwork()
    {
        for(int input=0; input< this.noOfInputNeuron; input++)
        {
            for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
            {
                this.weightFromInputToHidden[input][hidden] = getRandom();
            }
        }
        for(int hidden =0 ; hidden < this.noOfHiddenNeuron; hidden ++)
        {
            for(int output = 0; output< this.noOfOutputNeuron; output ++)
            {
                this.weightFromHiddenToOutput[hidden][output] = getRandom();
            }
        }
        //Intialize bias
        for(int b=0; b < this.noOfHiddenNeuron; b++)
        {
            this.biasForHiddenNode[b] = getRandom();
        }
        for(int b=0; b < this.noOfOutputNeuron; b++)
        {
            this.biasForOutputNode[b] = getRandom();
        }
    }
   
    public void addTrainningData(float[] inputVector,float[] targetOutput)
    {
        this.inputPatterns.add(inputVector);
        this.targetValues.add(targetOutput);
    }

    void setInput()
    {
        
        //Assign input from inputpattern to inputlayer

        //Get random Input pattern
        int random =(int) ( Math.random() * (this.indexOfPattern.size()) ) % this.indexOfPattern.size();
        this.currentInputPattern = this.indexOfPattern.get(random);
        this.indexOfPattern.remove(random);
        this.inputNeuron = (float[])this.inputPatterns.get(this.currentInputPattern);
        this.actualTargetValues = (float[])this.targetValues.get(this.currentInputPattern);
        /////System.out.println("Current Input Pattern " + this.currentInputPattern );
        if(currentInputPattern + 1 >= this.inputPatterns.size())
        {
            currentInputPattern = 0;
        }
        else
        {
            currentInputPattern ++;
        }

    }
    void feedForward()
    {
       //************ HIDDEN LAYER **********************
        //First ClearOut
        for(int h=0; h < this.noOfHiddenNeuron; h++)
        {
            this.hiddenNeuron[h]= 0.F;
        }
        //For Hidden Layer
        for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            for(int input=0; input < this.noOfInputNeuron; input++)
            {
                //Do part of summation, algorithm line 6
                this.hiddenNeuron[hidden] += this.inputNeuron[input]* this.weightFromInputToHidden[input][hidden];
            }
        }
        //Add bias
        for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            this.hiddenNeuron[hidden]+= this.biasForHiddenNode[hidden];
        }
        //Calculate sigmod for hidden layer, to calculate output
        for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            //Calculate output of each hidden neuron
            this.hiddenNeuron[hidden] = sigmod( this.hiddenNeuron[ hidden ]);
        }

        //*********** Output Layer ********************
        //First clear out
        for(int output=0; output < this.noOfOutputNeuron; output++)
        {
            this.outputNeuron[output] = 0.F;
        }
        for(int output=0; output < this.noOfOutputNeuron; output ++)
        {
            for(int hidden =0 ; hidden < this.noOfHiddenNeuron; hidden ++)
            {
                this.outputNeuron[output] += this.hiddenNeuron[ hidden] * this.weightFromHiddenToOutput[hidden][output];
            }
        }
        //Add bias for output layer
        for(int output =0; output < this.noOfOutputNeuron; output ++)
        {
            this.outputNeuron[ output ] += this.biasForOutputNode[output];
        }
        //Compute the actual output of output layer
        for(int output=0; output < this.noOfOutputNeuron; output ++)
        {
            this.outputNeuron[ output ] = sigmod( this.outputNeuron[ output ]);
        }
        
    }
    public static float sigmod(float value)
    {
        return (float) (1.0f / (1.0f + Math.exp((double) (-value))));
    }
    float backPropagate()
    {

        this.meanError = 0;
        //Calculate error for output layer
        for(int output =0 ; output < this.noOfOutputNeuron; output ++)
        {
            this.errorOfOutputLayer[ output ] = this.outputNeuron[output] * (1- this.outputNeuron[output]) * (this.actualTargetValues[output]- this.outputNeuron[output]);
            //this.meanSquareError[ output ] = this.
            this.meanError += Math.pow( this.actualTargetValues[ output ] - this.outputNeuron[ output ],2);
        }
        //Compute error for hidden layer
        //hidden is j , output is k
        for(int hidden =0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            this.errorOfHiddenLayer[hidden] = 0;
            for(int output =0; output< this.noOfOutputNeuron; output++)
            {
                //Compute summation part first
                this.errorOfHiddenLayer[hidden] += this.errorOfOutputLayer[output]* this.weightFromHiddenToOutput[hidden][output];
            }
        }
        //Compute error for each of unit in hidden layer

        for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            this.errorOfHiddenLayer[hidden] *= this.hiddenNeuron[hidden]*  (1- this.hiddenNeuron[hidden]);
        }

        //Weight Update for hidden to output layer
        for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            for( int output = 0; output < this.noOfOutputNeuron; output ++ )
            {
                float deltaWeight = this.learningRate * this.errorOfOutputLayer[output] * this.hiddenNeuron[hidden];
                this.weightFromHiddenToOutput[hidden][output] += deltaWeight;
            }
        }
        //Weight Update for input to hidden layer
        for(int input =0; input < this.noOfInputNeuron; input++)
        {
            for(int hidden =0; hidden < this.noOfHiddenNeuron; hidden ++)
            {
                float deltaWeight = this.learningRate * this.errorOfHiddenLayer[hidden ] * this.inputNeuron[input];
                float before = this.weightFromInputToHidden[input][hidden];
                this.weightFromInputToHidden[input][hidden]+= deltaWeight;
                ///System.out.println("Change from " + before +" to " + this.weightFromInputToHidden[input][hidden] + " delta is " + deltaWeight);
            }
        }
        //Bias update for output layer
        for(int output=0; output< this.noOfOutputNeuron; output++)
        {
            float deltaBias = this.learningRate* this.errorOfOutputLayer[output];
            this.biasForOutputNode[output] += deltaBias;
        }
        for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            float deltaBias = this.learningRate * this.errorOfHiddenLayer[hidden];
            this.biasForHiddenNode[ hidden ] += deltaBias;
        }
        float sum = 0F;
     
        return this.meanError;

    }

    int getFiredNeuronIndex()
    {
        int max = 0;

        for(int i=0; i < this.noOfOutputNeuron; i++)
        {
            if( this.outputNeuron[i] > this.outputNeuron[max])
            {
                max = i;
            }
        }
        return max;
    }
    float getLearningRate()
    {
        return this.learningRate;
    }
    float getHiddenNode()
    {
        return this.noOfHiddenNeuron;
    }
    float getMeanSquareError()
    {
        return this.meanError;
    }
    float getAccuracy()
    {
        return this.accuracy;
    }
    public void recall(float[] input)
    {
           this.inputNeuron = input;
           this.feedForward();

           System.out.println("Recall ");
           
           for(int out= 0; out < this.noOfOutputNeuron; out ++)
           {
               System.out.print(this.outputNeuron[out] + "  ");

           }
           
           System.out.println("");

    }
    double trainAnEpcho()
    {
        this.indexOfPattern = new ArrayList<Integer>();
        for(int i=0;i < this.inputPatterns.size();i++)
        {
            this.indexOfPattern.add(i);
        }
        
        //error is Mean Square Error
        float error = 0 ;

        for(int i=0; i < this.inputPatterns.size();i++)
        {
             setInput();
             feedForward();
             error  += backPropagate();
        }
        error /= (this.inputPatterns.size());
        this.meanError = error;
        return this.meanError;
    }
    
    //Return accuracy on
    //Parameter are input pattern and target output age
    float testOnDataSet(Vector testSet,ArrayList<Integer> correctAge)
    {
        accuracy = 0F;
        float correct = 0;

        for(int i=0; i < testSet.size();i ++)
        {
            float[] inputPattern = (float[]) testSet.get(i);
            this.recall(inputPattern);

            int firedNeuronIndex = this.getFiredNeuronIndex();


            int outputAge = firedNeuronIndex + 1;
            int cAge = correctAge.get(i);

            if( outputAge == cAge )
            {
                correct ++;
            }
        }
        accuracy = (correct/ testSet.size() )*100;
        ///System.out.println("All test size "+ testSet.size() + "  Correctly classified "+ correct + " Accuracy is " + accuracy);
        //recall and get age
        
        return accuracy ;
    }

    public void showWeightAndBias()
    {
        System.out.println("Input to Hidden");
        for(int input =0; input < this.noOfInputNeuron; input++)
        {
            for(int hidden =0; hidden < this.noOfHiddenNeuron; hidden ++)
            {
                System.out.println("Weight Input "+input+" hidden "+ hidden+" => "+ this.weightFromInputToHidden[input][hidden] );
                
            }
        }
        System.out.println("Hidden to output ");
         for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            for( int output = 0; output < this.noOfOutputNeuron; output ++ )
            {
                System.out.println("Weight hidden "+hidden+" hidden "+ output+ " => "+this.weightFromHiddenToOutput[hidden][output] );
            }
        }
        System.out.println("Bias of hidden");
        for(int hidden=0; hidden < this.noOfHiddenNeuron; hidden++)
        {
            System.out.println("Bias of hidden "+ hidden +" => "+ this.biasForHiddenNode[ hidden ]);
           
        }
        System.out.println("Bias of output ");
        for(int output=0; output< this.noOfOutputNeuron; output++)
        {
            System.out.println("Bias of hidden "+ output +" => "+ this.biasForOutputNode[output]);
        
        }
        
    }
    
    public static void main(String[]args)
    {
        FeedForwardNeuralNetwork neural = new FeedForwardNeuralNetwork(2, 2, 2);
        neural.setLearningRate(0.03F);

        neural.addTrainningData(new float[]{0,0},new float[]{1,0});
        neural.addTrainningData(new float[]{0,1},new float[]{0,1});
        neural.addTrainningData(new float[]{1,0},new float[]{0,1});
        neural.addTrainningData(new float[]{1,1},new float[]{1,0});

        double error = 0;
        do
        {
            error = neural.trainAnEpcho();
            System.out.println("MeanSquare " + error);
        }while(error > 0.1);
        neural.showWeightAndBias();
        neural.recall(new float[]{0,0});
        neural.recall(new float[]{0,1});
        neural.recall(new float[]{1,0});
        neural.recall(new float[]{1,1});


      
    }
    
}
