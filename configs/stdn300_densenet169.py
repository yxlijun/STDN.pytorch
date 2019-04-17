model = dict(
    type='STDN',
    input_size=300,
    init_net=True,
    rgb_means=(104, 117, 123),
    voc_config=dict(
        num_classes=21,
        backbone='densenet',
        predict_channel=[104, 360, 1280, 1120, 960, 800],
        anchor_number=[8, 8, 8, 8, 8, 8],
        scale_param=[9, 3, 2, 2, 4]
    ),
    coco_config=dict(
        num_classes=81,
        backbone='densenet',
        predict_channel=[104, 360, 1280, 1120, 960, 800],
        anchor_number=[8, 8, 8, 8, 8, 8],
        scale_param=[9, 3, 2, 2, 4]
    ),
    p=0.6,
    anchor_config=dict(
        feature_maps=[36, 18, 9, 5, 3, 1],
        step_pattern=[8, 16, 33, 75, 100, 300],
        size_pattern=[0.08, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
    ),
    save_epochs=10,
    weights_save='weights/'
)

train_cfg = dict(
    cuda=True,
    warmup=5,
    per_batch_size=16,
    lr=[1e-3, 1e-4, 1e-5],
    gamma=0.1,
    end_lr=1e-6,
    step_lr=dict(
        COCO=[90, 110, 130, 150],
        VOC=[500, 600, 700, 750],
    ),
    print_epochs=10,
    num_workers=8,
)

test_cfg = dict(
    cuda=True,
    topk=0,
    iou=0.45,
    soft_nms=True,
    score_threshold=0.1,
    keep_per_class=50,
    save_folder='eval',
)

loss = dict(overlap_thresh=0.5,
            prior_for_matching=True,
            bkg_label=0,
            neg_mining=True,
            neg_pos=3,
            neg_overlap=0.5,
            encode_target=False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC=dict(
        train_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets=[('2007', 'test')],
    ),
    COCO=dict(
        train_sets=[('2014', 'train'), ('2014', 'valminusminival')],
        eval_sets=[('2014', 'minival')],
        test_sets=[('2015', 'test-dev')],
    )
)

import os
home = os.path.expanduser("~")
VOCroot = os.path.join(home, "data/VOCdevkit/")
COCOroot = os.path.join(home, "data/coco/")
