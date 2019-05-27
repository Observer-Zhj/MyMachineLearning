package cn.zhj.deeplearning;

import java.util.ArrayList;

public class OutputLayer extends Layer {

    public OutputLayer() {}

    public OutputLayer(int cellNum) {
        for (int i = 0; i < cellNum; i++) {
            BasicCell cell = new HiddenCell();
            addCell(cell);
        }
    }

    public OutputLayer(int cellNum, String activeFunction) {
        for (int i = 0; i < cellNum; i++) {
            BasicCell cell = new HiddenCell(activeFunction);
            addCell(cell);
        }
    }

    public OutputLayer(ArrayList<OutputCell> cellList) {
        this.cellList = new ArrayList<BasicCell>(cellList);
    }

    @Override
    public void fullyConnect(Layer layer) {}
}
