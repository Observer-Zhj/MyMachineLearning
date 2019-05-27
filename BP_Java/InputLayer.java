package cn.zhj.deeplearning;

import java.util.ArrayList;

public class InputLayer extends Layer {

    public InputLayer() {}

    public InputLayer(int cellNum) {
        for (int i = 0; i < cellNum; i++) {
            BasicCell cell = new InputCell();
            addCell(cell);
        }
    }

    public InputLayer(int cellNum, double[] value) {
        for (int i = 0; i < cellNum; i++) {
            BasicCell cell = new InputCell(value[i]);
            addCell(cell);
        }
    }

    public InputLayer(ArrayList<InputCell> cellList) {
        this.cellList = new ArrayList<BasicCell>(cellList);
    }

    @Override
    public void calculateDelta() {}

}
