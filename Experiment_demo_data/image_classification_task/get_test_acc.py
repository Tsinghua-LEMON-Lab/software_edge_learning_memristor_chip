import numpy as np


def get_weights(w1_path, w2_path):
    W1 = np.loadtxt(w1_path, str, delimiter=",", usecols=range(200))
    W1 = np.float64(W1)
    w1_plus = W1[:, ::2].T
    w1_minus = W1[:, 1::2].T
    Weight1 = w1_plus - w1_minus
    Weight1 /= np.max(Weight1)

    W2 = np.loadtxt(w2_path, str, delimiter=",", usecols=range(20))
    W2 = np.float64(W2)
    w2_plus = W2[:, 1::2].T
    w2_minus = W2[:, ::2].T
    Weight2 = w2_plus - w2_minus
    Weight2 /= np.max(Weight2)
    Weight2[9, :] = np.random.randn(100) * 0.2

    return Weight1, Weight2


def get_data_label(sourcetestdata_path,
                   sourcetestlabel_path,
                   targettestdata_path,
                   targettestlabel_path,
                   ):
    TestSourceData = np.load(sourcetestdata_path)
    TestSourceLabel_onehot = np.load(sourcetestlabel_path)
    TestSourceData_add = np.zeros((TestSourceData.shape[0], 784 - TestSourceData.shape[1]))
    TestSourceData0 = np.hstack((TestSourceData, TestSourceData_add))
    TestSourceLabel0 = np.argmax(TestSourceLabel_onehot, axis=1)
    TestData = np.load(targettestdata_path)
    TestLabel_onehot = np.load(targettestlabel_path)
    TestData_add = np.zeros((TestData.shape[0], 784 - TestData.shape[1]))
    TestData = np.hstack((TestData, TestData_add))
    TestLabel = np.argmax(TestLabel_onehot, axis=1)

    return TestSourceData0, TestSourceLabel0, TestData, TestLabel


def relu_activation(x, thresh=0):
    return np.maximum(0, x - thresh)


def get_acc(Weight1, Weight2, TestData, TestLabel):
    accuracy_cls_list = []
    pre_cls_list = []
    test_image_num = TestData.shape[0]
    Z1 = np.dot(Weight1, TestData.T)
    X1_out = relu_activation(Z1)
    Z2 = np.dot(Weight2, X1_out)
    X2_out = Z2.T
    pre = np.argmax(X2_out, axis=1)
    accuracy = np.sum(pre == TestLabel.flatten()) / float(test_image_num)
    for i in range(10):
        sigle_cls = TestLabel[TestLabel == i]
        pre_cls = np.argmax(X2_out[TestLabel == i], axis=1)
        accuracy_cls = np.sum(pre_cls == sigle_cls.flatten()) / float(sigle_cls.shape[0] + 1e-15)
        accuracy_cls_list.append(accuracy_cls)
        pre_cls_list.append(sigle_cls.shape[0])
    return accuracy, accuracy_cls_list, pre_cls_list


if __name__ == '__main__':
    Weight1, Weight2 = get_weights(w1_path='weights/w1.txt', w2_path='weights/w2.txt')

    TestSourceData0, TestSourceLabel0, TestData, TestLabel = get_data_label(
        sourcetestdata_path='datasets/sourcetestdata.npy',
        sourcetestlabel_path='datasets/sourcetestlabel.npy',
        targettestdata_path='datasets/targettestdata.npy',
        targettestlabel_path='datasets/targettestlabel.npy',
    )
    source_accuracy, _, _ = get_acc(Weight1, Weight2, TestSourceData0, TestSourceLabel0)
    print('source test acc %.4f' % (source_accuracy))

    target_accuracy, _, _ = get_acc(Weight1, Weight2, TestData, TestLabel)
    print('target test acc %.4f' % (target_accuracy))
