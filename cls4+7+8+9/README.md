## Classification/Retreival Results

### Food Classification(Task 4)

```
nsml run -d 4_cls_food -g 1 --memory 24G --shm-size 24G --cpus 4 -e main.py -a "-c confs/resnet50.yaml --cutmix-prob 0.5 --cv 0" --memo "resnet50 lr=0.001 adam cutmix=0.5 cv=0"
```

|                                               | cv1    | cv2    | cv3    | Avg    | LB     |
|-----------------------------------------------|-------:|-------:|-------:|-------:|-------:|
| Resnet18 lr=0.001 ADAM epoch@100 158 resave   | 0.7693 |        |        |        | 0.6896 |
| Resnet50 lr=0.001 ADAM epoch@95  89 transfer  | 0.7787 |        |        |        | 0.7705 |
| Cutmix 0.5 cv1                   103 100      |        | 0.8579 |        |        | 0.7917 |
| Cutmix 0.5 cv2                   162 100      |        |        | 0.8579 |        | 0.7805 |
| Ensemble cv1(1.0) + cv2(0.5)                  |        |        |        |        | 0.8046 |
| Ensemble cv1(1.0) + cv2(0.6)                  |        |        |        |        | 0.8056 |

* team_286/4_cls_food/30 95
* team_286/4_cls_food/89 transfer : 0.7498 ??

### Face Classification(Task 7)

```
nsml run -d 7_icls_face -g 1 --memory 24G --shm-size 24G --cpus 4 -e main.py -a "-c confs/resnet50.yaml --cv 0" --memo "resnet50 lr=0.001 adam cv=0"
```

|                                         | cv1    | LB     |
|-----------------------------------------|-------:|-------:|
| Resnet50 lr=0.001 ADAM epoch@10         | 0.9233 | 0.3329 |
| Resnet50 lr=0.001 ADAM epoch@55         | 0.9566 | 0.4173 |
| + only target class                     | 0.9566 | 0.8662 |
| Resnet50 lr=0.001 ADAM epoch@95         | 0.9566 | 0.9632 |


### Food Image Retreival(Task8)

|                                         | cv1    | LB     |
|-----------------------------------------|-------:|-------:|
| Resnet50 lr=0.001 ADAM epoch@95         | 0.7787 | 0.7214 |
| + using logit                           | 0.7787 | 0.7107 |
| Cutmix 0.5 cv1             103 100      | 0.8579 | 0.7728 |

* team_286/4_cls_food/30 95
* team_286/4_cls_food/89 transfer

### Car Image Retreival(Task9)

|                                         | cv1    | LB     |
|-----------------------------------------|-------:|-------:|
| Resnet50 lr=0.001 ADAM epoch@10         | 0.9459 | 0.8271 |
| Resnet50 lr=0.001 ADAM epoch@20         | 0.9739 | 0.8715 |
| + using logit                           |        | 0.7956 |

* team_286/9_iret_car/16 20
