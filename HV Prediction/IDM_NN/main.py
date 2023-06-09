import argparse
import train
import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platoon_num', type=int, help='Platoon number')
    args = parser.parse_args()
    platoon_num = args.platoon_num

    platoon_num = 1

    # 设置look_back和其它参数
    look_back = 40
    look_forward = 1
    epochs = 50
    batch_size = 64

    # 训练模型
    train.train_model(platoon_num, look_back, epochs, batch_size)

    # 预测并绘制结果图
    predict.predict_results(platoon_num, look_back)
