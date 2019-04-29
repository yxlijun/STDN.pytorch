model = dict(
    type='STDN',
    input_size=513,
    init_net=True,
    rgb_means=(104, 117, 123),
    voc_config=dict(
        num_classes=21,
        backbone='densenet',
        predict_channel=[104, 360, 1280, 1120, 960, 800],
        anchor_number=[8, 8, 8, 8, 8, 8],
        scale_param=[16, 3, 2, 2, 4]
    ),
    coco_config=dict(
        num_classes=81,
        backbone='densenet',
        predict_channel=[104, 360, 1280, 1120, 960, 800],
        anchor_number=[8, 8, 8, 8, 8, 8],
        scale_param=[16, 3, 2, 2, 4]
    ),
    p=0.6,
    anchor_config=dict(
        feature_maps=[64, 32, 16, 8, 5, 1],
        steps=[8, 16, 32, 64, 102, 513],
        aspect_ratios=[[1.6, 2, 3], [1.6, 2, 3], [1.6, 2, 3],
                       [1.6, 2, 3], [1.6, 2, 3], [1.6, 2, 3]],
        VOC=dict(
            min_ratio=20,
            max_ratio=90
        ),
        COCO=dict(
            min_ratio=15,
            max_ratio=90
        )
    ),
    save_epochs=10,
    pretained_model='weights/densenet169.pth',
    weights_save='weights/'
)

train_cfg = dict(
    cuda=True,
    per_batch_size=16,
    lr=1e-2,
    gamma=-0.9,
    end_lr=1e-6,
    step_lr=dict(
       COCO=[90, 200, 300, 350],
        VOC=[500, 650, 750, 800],
    ),
    print_epochs=10,
    num_workers=8,
)

test_cfg = dict(
    cuda=True,
    topk=0,
    iou=0.45,
    soft_nms=True,
    score_threshold=0.01,
    keep_per_class=200,
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
