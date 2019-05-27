package cn.zhj.deeplearning;

public class OutputCell extends BasicCell{


    public OutputCell() {
        activeFunction = null;
    }

    public OutputCell(String activeFunction) {
        this.activeFunction = activeFunction;
    }


}
