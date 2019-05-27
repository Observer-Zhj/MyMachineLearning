package cn.zhj.deeplearning;

import java.util.ArrayList;
import java.util.List;

/**
 * 输入神经元，成员变量inpValue接收输入值，nextLayer储存下一层神经元的引用，是个ArrayList
 * 成员方法setValue修改inpValue的值。
 * 成员方法setNextLayer修改下一层神经元。
 * 成员方法addNextCell添加后连的神经元的引用到成员变量nextLayer中
 * 成员方法addNextLayer增加一层神经元的引用到成员变量nextLayer中，不是替换nextLayer
 *
 */
public class InputCell extends BasicCell{


    protected InputCell(){}

    /** 输入神经元的输出值跟输入值相等 */
    public InputCell(double inpValue) {
        this();
        this.inpValue = this.outValue = inpValue;
    }

    @Override
    public void setInpValue(double inpValue) {
        this.inpValue = this.outValue = inpValue;
    }

    /** 输入神经元没有激活函数 */
    @Override
    public void setActiveFunction(String activeFunction) {}

    public static void main(String[] args) {

    }
}
