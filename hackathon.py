import os
import random
import time

import mxnet as mx
import mxnet.ndarray as nd

from mxnet.gluon.model_zoo.vision import mobilenet1_0
from mxnet import init, image, autograd, nd
from mxnet.image import color_normalize
from mxnet.gluon.data.vision import ImageRecordDataset
import mxnet.gluon as gluon

pretrained_net = mobilenet1_0(pretrained=True)
net = mobilenet1_0(classes=2)

net.features = pretrained_net.features
net.output.initialize(init.Xavier())

license_map = {
    'd': [],
    'n': []
}

license_to_idx = {
    'n': 0,
    'd': 1
}

data_dir = "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\licenses mxnet"
fnames = os.listdir(data_dir)
counter = 0
for fn in fnames:
    arr = fn.split('_')
    category = arr[0]
    license_map[category].append({
        'idx': counter,
        'label': license_to_idx[category],
        'filename': fn
    })
    counter += 1

random.shuffle(license_map['n'])
random.shuffle(license_map['d'])

def write_lst(image_arr, base_dir, file_path):
    with open(file_path, 'w') as f:
        count = 0
        for img in image_arr:
            label = img['label']
            img_path = os.path.join(base_dir, img['filename'])
            new_line = '\t'.join([str(count), str(label), str(img_path)])
            new_line += '\n'
            f.write(new_line)
            count += 1

min_data_len = min(len(license_map['n']), len(license_map['d']))
sample = (0, 8)
train = (0, int(min_data_len * 0.8))
validation = (int(min_data_len * 0.8), int(min_data_len * 0.9))
test = (int(min_data_len * 0.9), int(min_data_len * 1))

def split_dataset(from_idx, to_idx):
    return license_map['n'][from_idx: to_idx] + license_map['d'][from_idx: to_idx]

sample_set = split_dataset(sample[0], sample[1])
write_lst(sample_set, "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\licenses mxnet",
          "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\license_sample.lst")

train_set = split_dataset(train[0], train[1])
write_lst(train_set, "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\licenses mxnet",
          "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\license_train.lst")

validation_set = split_dataset(validation[0], validation[1])
write_lst(validation_set, "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\licenses mxnet",
          "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\license_validation.lst")

test_set = split_dataset(test[0], test[1])
write_lst(validation_set, "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\licenses mxnet",
          "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon\\license_test.lst")



train_augs = [
    image.ResizeAug(224),
    image.HorizontalFlipAug(0.6),
    image.BrightnessJitterAug(0.3),
    image.HueJitterAug(0.1)]

test_augs = [image.ResizeAug(224)]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2, 0, 1))
    return data, nd.array([label]).asscalar().astype('float32')

train_rec = "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\Parking Space Detector MXNet\\images\\train\\car_train.rec"
validation_rec = "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\Parking Space Detector MXNet\\images\\val\\car_val.rec"
test_rec = "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\Parking Space Detector MXNet\\images\\test\\car_test.rec"
sample_rec = "C:\\Users\\Jason\\Documents\\Python Projects\\ml\\Parking Space Detector MXNet\\images\\sample\\car_sample.rec"

trainIterator = ImageRecordDataset(
    filename = train_rec,
    transform = lambda X, y: transform(X, y, train_augs)
)
validationIterator = ImageRecordDataset(
    filename = validation_rec,
    transform = lambda X, y: transform(X, y, test_augs)
)
testIterator = ImageRecordDataset(
    filename = test_rec,
    transform = lambda X, y: transform(X, y, test_augs)
)
sampleIterator = ImageRecordDataset(
    filename = sample_rec,
    transform = lambda X, y: transform(X, y, test_augs)
)

def train(net, ctx,
          batch_size = 4, epochs = 10, learning_rate = 0.01, wd = 0.001):
    train_data = gluon.data.DataLoader(
        trainIterator, batch_size, shuffle=True)
    validation_data = gluon.data.DataLoader(
        validationIterator, batch_size)

    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})

    train_util(net, train_data, validation_data,
               loss, trainer, ctx, epochs, batch_size)

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        data = color_normalize(data/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
        output = net(data)
        prediction = nd.argmax(output, axis=1)
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]

def train_util(net, train_iter, validation_iter, loss_fn, trainer, ctx, epochs, batch_size):
    metric = mx.metric.create(['acc'])
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_iter):
            st = time.time()
            # ensure context
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # normalize images
            data = color_normalize(data / 255,
                                   mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)),
                                   std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)))

            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)

            loss.backward()
            trainer.step(data.shape[0])

            #  Keep a moving average of the losses
            metric.update([label], [output])
            names, accs = metric.get()
            # print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(epoch, i, batch_size/(time.time()-st), metric_str(names, accs)))
            #if i % 100 == 0:
                #net.collect_params().save('C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon' % (str(epoch), str(i)))

        train_acc = evaluate_accuracy(train_iter, net)
        validation_acc = evaluate_accuracy(validation_iter, net)
        print("Epoch %s | training_acc %s | val_acc %s " % (epoch, train_acc, validation_acc))


def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])

ctx = mx.cpu()
train(net, ctx, batch_size = 10, epochs = 5, learning_rate = 0.0005)

#net.export("C:\\Users\\Jason\\Documents\\Python Projects\\ml\\hackathon", epoch = 5)


test_data_loader = gluon.data.DataLoader(testIterator, 8)
test_acc = evaluate_accuracy(test_data_loader, net)
print(test_acc)