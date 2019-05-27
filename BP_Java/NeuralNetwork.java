package cn.zhj.deeplearning;

import java.util.ArrayList;

/**
 * Created by admin on 2018/10/18.
 */
public class NeuralNetwork {

    public static void forward(ArrayList<Layer> nn, double[] x) {
        int l = nn.size();
        nn.get(0).setInpValue(x);
        for (int i = 1; i < l; i++) {
            nn.get(i).forward();
        }
    }

    public static double calCost(Layer layer, double[] y) {

        double[] output = layer.getOutValue();
        double cost = CostFunc.squareCostOneSample(output, y);
        return cost;
    }

    public static void backPropagation(ArrayList<Layer> nn, double[] y) {
        int l = nn.size();
        nn.get(l-1).calculateOutCellDelta(y);
        for (int i = l-2; i > 0; i--) {
            nn.get(i).calculateDelta();
        }
        for (int i = 1; i < l; i++) {
            nn.get(i).update(0.5);
        }
    }


    public static void main(String[] args) {
        int[] size = {2, 3, 1};
        double[][] inpValue = {{1,1}, {0,0}, {1,0}, {0,1}};
        double[][] y = {{0}, {0}, {1}, {1}};
        // 创建网络
        ArrayList<Layer> nn = new ArrayList<Layer>();
        // 创建输入层
        Layer inpLay = new InputLayer(size[0]);
        nn.add(inpLay);
        // 连接隐含层和输出层
        for (int i = 1; i < size.length; i++) {
            Layer hidLay = new HiddenLayer(size[i]);
            nn.get(i-1).fullyConnect(hidLay);
            nn.add(hidLay);
        }

        java.util.Date dt1 = new java.util.Date();
        long t1 = dt1.getTime();
        Layer outLay = nn.get(2);
        for (int i = 0; i < 100000; i++) {
            for (int j = 0; j < 4; j++) {
                forward(nn, inpValue[j]);
                backPropagation(nn, y[j]);
            }
        }
        java.util.Date dt2 = new java.util.Date();
        long t2 = dt2.getTime();
        System.out.println("花费" + (double) ((t2-t1))/1000 + "秒");
        forward(nn, inpValue[0]);
        System.out.println(outLay.getOutValue()[0]);
        forward(nn, inpValue[1]);
        System.out.println(outLay.getOutValue()[0]);
        forward(nn, inpValue[2]);
        System.out.println(outLay.getOutValue()[0]);
        forward(nn, inpValue[3]);
        System.out.println(outLay.getOutValue()[0]);


    }
}
