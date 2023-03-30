def hue_calculate(round1, round2, delta, add_num):
    return (((round1 - round2) / delta) * 60 + add_num) % 360


def rgb_to_hsv(rgb_seq):
    r, g, b = rgb_seq
    r_round = float(r) / 255
    g_round = float(g) / 255
    b_round = float(b) / 255
    max_c = max(r_round, g_round, b_round)
    min_c = min(r_round, g_round, b_round)
    delta = max_c - min_c

    h = None
    if delta == 0:
        h = 0
    elif max_c == r_round:
        h = hue_calculate(g_round, b_round, delta, 360)
    elif max_c == g_round:
        h = hue_calculate(b_round, r_round, delta, 120)
    elif max_c == b_round:
        h = hue_calculate(r_round, g_round, delta, 240)
    if max_c == 0:
        s = 0
    else:
        s = (delta / max_c) * 100
    v = max_c * 100
    return h, s, v

def find_color_series(rgb_seq):  # TODO:此处是否有更好实现？
    """
    将rgb转为hsv之后根据h和v寻找色系
    :param rgb_seq:
    :return:
    """
    h, s, v = rgb_to_hsv(rgb_seq)
    cs = None
    if 30 < h <= 90:
        cs = 'yellow'
    elif 90 < h <= 150:
        cs = 'green'
    elif 150 < h <= 210:
        cs = 'cyan'
    elif 210 < h <= 270:
        cs = 'blue'
    elif 270 < h <= 330:
        cs = 'purple'
    elif h > 330 or h <= 30:
        cs = 'red'

    if s < 10:  # 色相太淡时，显示什么颜色主要由亮度来决定
        cs = update_by_value(v)
    return cs
    
def update_by_value(v):
    """
    根据 V 值去更新色系数据
    :param v: 
    :return: 
    """
    if v <= 100 / 3 * 1:
        cs = 'black'
    elif v <= 100 / 3 * 2:
        cs = 'gray'
    else:
        cs = 'white'
    return cs


if __name__ == '__main__':

    color_list = [[128,127,255], [255,123,251]]

    for item in color_list:
        print(find_color_series(item))
