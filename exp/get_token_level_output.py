#記事ごとにf値を出すプログラム
def read_pred():
  from sklearn.metrics import classification_report
  f = open('1.out', 'r')

  y_true = []
  y_pred = []

  datalist = f.read().splitlines()

  for data in datalist:
    list = data.split()
    if 'DOCID' in list[0]:
      pass
    else: 
      if list[8] == 'O':
        y_true.append(0)
      else:
        y_true.append(1)
      if list[9] == 'O':
        y_pred.append(0)
      else:
        y_pred.append(1)
  return y_pred, y_true