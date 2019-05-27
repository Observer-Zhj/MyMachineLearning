package cn.zhj.deeplearning;

import java.util.ArrayList;

public abstract class Layer {
    ArrayList<BasicCell> cellList = new ArrayList<BasicCell>();

    /** 添加一个神经元 */
    public void addCell(BasicCell cell) {
        cellList.add(cell);
    }

    /** 添加几个神经元 */
    public void addCell(ArrayList<BasicCell> cellArrayList) {
        cellList.addAll(cellArrayList);
    }

    /** 向后全连接一层神经元 */
    public void fullyConnect(Layer layer) {
        for (BasicCell cell : cellList) {
            cell.addNextLayer(layer.cellList);
        }
    }

    public void setInpValue(double[] inpValue) {
        if (inpValue.length != cellList.size()) {
            throw new IllegalArgumentException("输入值数量与神经元数量不相等");
        }
        for (int i = 0; i < inpValue.length; i++) {
            cellList.get(i).setInpValue(inpValue[i]);
        }
    }

    /** 计算这层神经元的输入值 */
    public double[] getInpValue() {
        double[] inpValue = new double[cellList.size()];
        for (int i = 0; i < cellList.size(); i++) {
            inpValue[i] = cellList.get(i).getInpValue();
        }
        return inpValue;
    }

    /** 计算这层神经元的输出值 */
    public double[] getOutValue() {
        double[] outValue = new double[cellList.size()];
        for (int i = 0; i < cellList.size(); i++) {
            outValue[i] = cellList.get(i).getOutValue();
        }
        return outValue;
    }

    public void calculateInpValue() {
        for (BasicCell cell : cellList) {
            cell.calculateInpValue();
        }
    }

    public void calculateOutValue() {
        for (BasicCell cell : cellList) {
            cell.calculateOutValue();
        }
    }

    /** 前馈，计算这层神经元的输入值和输出值 */
    public void forward() {
        for (BasicCell cell : cellList) {
            cell.calculateInpValue();
            cell.calculateOutValue();
        }
    }

    /**
     * 用反向传播算法计算delta。
     * 需要注意的是，计算的是上一层每个神经元的delta，和这一层的每个神经元的deltaWeights，deltaBias。
     */
    public void calculateDelta() {
        for (BasicCell cell : cellList) {
            cell.calculateDelta();
        }
    }

    public void calculateOutCellDelta(double[] y) {
        for (int i = 0; i < cellList.size(); i++) {
            cellList.get(i).calculateOutCellDelta(y[i]);
        }
    }

    /** 用梯度下降法（SGD）这一层更新这一层所有神经元的权重和偏置 */
    public void update(double rate, int batch) {
        for (BasicCell cell : cellList) {
            cell.update(rate, batch);
        }
    }

    public void update() {
        update(0.1, 1);
    }

    public void update(double rate) {
        update(rate, 1);
    }

    public void update(int batch) {
        update(0.1, batch);
    }


}
