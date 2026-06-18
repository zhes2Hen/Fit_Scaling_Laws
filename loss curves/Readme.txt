请使用pandas工具包读取pkl文件，其中Metrics/loss列为对应的loss值， lr列表示对应的学习率。

注：其中的``8-1-1'' LRS表示按照总的training steps，在80%处，90%处，学习率都decay为原来上一阶段的1/\sqrt{10}，也即是假设最大学习率为\eta，那么总共的three-stage学习率就为：\eta, \eta/\sqrt{10}, \eta/10。