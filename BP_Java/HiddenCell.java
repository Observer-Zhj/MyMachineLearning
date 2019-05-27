package cn.zhj.deeplearning;

public class HiddenCell extends BasicCell{


    public HiddenCell() {
        activeFunction = "sigmoid";
    }

    public HiddenCell(String activeFunction) {
        this.activeFunction = activeFunction;
    }



}
