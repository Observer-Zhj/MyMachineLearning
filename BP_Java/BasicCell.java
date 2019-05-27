package cn.zhj.deeplearning;

import java.util.ArrayList;
import java.util.Random;

public abstract class BasicCell{
    double inpValue;
    double outValue;
    ArrayList<BasicCell> previousLayer = new ArrayList<BasicCell>();
    ArrayList<BasicCell> nextLayer = new ArrayList<BasicCell>();
    String activeFunction;
    double bias = 0;
    ArrayList<Double> weights = new ArrayList<Double>();
    double delta = 0;
    ArrayList<Double> deltaWeights = new ArrayList<Double>();
    double deltaBias = 0;



    public double getInpValue() {
        return inpValue;
    }

    public void setInpValue(double inpValue) {
        this.inpValue = inpValue;
    }

    public double getOutValue() {
        return outValue;
    }

    public double getBias() {
        return bias;
    }

    public ArrayList<Double> getWeights() {
        return weights;
    }

    public String getActiveFunction() {
        return activeFunction;
    }

    public void setActiveFunction(String activeFunction) {
        if (ActiveFunc.actFuncCollection.indexOf(activeFunction) < 0) {
            throw new IllegalArgumentException("激活函数不存在");
        }
        this.activeFunction = activeFunction;
    }

    public ArrayList<BasicCell> getNextLayer() {
        return nextLayer;
    }

    /** 神经元连接下一层 */
    public void setNextLayer(ArrayList<BasicCell> nextLayer) {
        this.nextLayer = new ArrayList<BasicCell>();
        addNextLayer(nextLayer);
    }

    public ArrayList<BasicCell> getPreviousLayer() {
        return previousLayer;
    }

    /** 计算神经元的输入值 */
    double calculateInpValue() {
        double result = bias;
        for (int i = 0; i < previousLayer.size(); i++) {
            result += previousLayer.get(i).outValue * weights.get(i);
        }
        inpValue = result;
        return inpValue;
    }

    /** 计算神经元的输出值，即输入值经过激活函数的值 */
    double calculateOutValue() {
        outValue = ActiveFunc.active(inpValue, activeFunction);
        return outValue;
    }

    /** 向后连接神经元 */
    public boolean addNextCell(BasicCell nextCell) {
        if (nextCell instanceof OutputCell)
            throw new IllegalArgumentException("输出神经元不能往后连接");
        this.nextLayer.add(nextCell);
        Random rnd = new Random();
        nextCell.weights.add(rnd.nextDouble() * 2 - 1);
        nextCell.deltaWeights.add(0.0);
        nextCell.previousLayer.add(this);
        return true;
    }

    /** 向前连接神经元 */
    public boolean addPreviousCell(BasicCell previousCell) {
        if (previousCell instanceof InputCell)
            throw new IllegalArgumentException("输入神经元不能往前连接");
        previousCell.addNextCell(this);
        return true;
    }

    /** 向后连接一层神经元 */
    public boolean addNextLayer(ArrayList<BasicCell> nextLayer) {
        for (BasicCell nextCell : nextLayer) {
            addNextCell(nextCell);
        }
        return true;
    }

    /** 计算用反向传播算法计算delta */
    void calculateDelta() {
        deltaBias = delta;
        for (int i = 0; i < previousLayer.size(); i++) {
            BasicCell cell = previousLayer.get(i);
            // 每一次BP后delta需要被初始化
            cell.delta += weights.get(i) * delta * ActiveFunc.diffActive(cell.inpValue, cell.activeFunction);
            // 每一个权重更新后deltaWeights和deltaBias需要初始化，即经过一个batch后重新初始化。
            deltaWeights.set(i, deltaWeights.get(i) + cell.outValue * delta);
        }
        // 这个神经元的BP结束，初始化为0
        delta = 0.0;
    }

    void calculateOutCellDelta(double y) {
        delta = (outValue - y) * ActiveFunc.diffActive(inpValue, activeFunction);
        calculateDelta();
    }

    /** 用梯度下降法（SGD）更新神经元的权重和偏置 */
    void update(double rate, int batch) {
        bias -= rate / batch * deltaBias;
        deltaBias = 0.0;
        for (int i = 0; i < deltaWeights.size(); i++) {
            double newWeight = weights.get(i) - rate / batch * deltaWeights.get(i);
            weights.set(i, newWeight);
            deltaWeights.set(i, 0.0);
        }
    }

    void update() {
        update(0.1, 1);
    }

    void update(double rate) {
        update(rate, 1);
    }

    void update(int batch) {
        update(0.1, batch);
    }

}
