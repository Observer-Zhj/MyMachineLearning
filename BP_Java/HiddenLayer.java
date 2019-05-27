package cn.zhj.deeplearning;

import java.util.ArrayList;

public class HiddenLayer extends Layer {

    public HiddenLayer() {}

    public HiddenLayer(int cellNum) {
        for (int i = 0; i < cellNum; i++) {
            BasicCell cell = new HiddenCell();
            addCell(cell);
        }
    }

    public HiddenLayer(int cellNum, String activeFunction) {
        for (int i = 0; i < cellNum; i++) {
            BasicCell cell = new HiddenCell(activeFunction);
            addCell(cell);
        }
    }

    public HiddenLayer(ArrayList<HiddenCell> cellList) {
        this.cellList = new ArrayList<BasicCell>(cellList);
    }
}
