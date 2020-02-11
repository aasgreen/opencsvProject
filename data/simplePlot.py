import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rand_plot(name):
    #np.random.seed(42)
    leading_degree = np.random.randint(low=1, high=5)
    poly = np.random.uniform(low=-1, high = 1, size=leading_degree)

    x_random = np.linspace(*np.random.random(1)*-1, *np.random.random(1), 100)
    #y_random = x_random+np.random.rand(100)*10 + np.random.rand(1)*400
    y_random = np.polyval(poly, x_random)
    y_random = y_random + y_random.max()/np.random.randint(100)*np.random.uniform(low=-1, high=1, size=100)

    fig, ax = plt.subplots(figsize=(4,4), dpi=100)
    ax.plot(x_random, y_random, '.')
    ax.set_ylabel(u'distance (\u00b5m)')
    ax.set_xlabel(u'time (s)')
    plt.tight_layout()
    fig.canvas.draw()
    red_point =[100,200]
    red_point = ax.transData.inverted().transform(red_point)
    ax.plot(red_point[0], red_point[1], '.')

    fig.savefig(name+'.png',dpi=100)
    #need to get and extract metadata for training from matplotlib objects
    x_tick_labels = ax.get_xticklabels()[1:-1]

    y_tick_labels = ax.get_yticklabels()[1:-1]
    meta_data = {
            'x_tick_text' : [tick.get_text() for tick in x_tick_labels],
            'x_tick_display_pos' : [tick._get_xy_display().tolist() for tick in x_tick_labels],
            'x_tick_window'      :[tick.get_window_extent().get_points().tolist() for tick in x_tick_labels],
            'y_tick_text' : [tick.get_text() for tick in y_tick_labels],
            'y_tick_display_pos' : [tick._get_xy_display().tolist() for tick in y_tick_labels],
                'y_tick_window'      :[tick.get_window_extent().get_points().tolist() for tick in y_tick_labels],
                'x'       : list(x_random),
                'y'       : list(y_random)
        }
	#create annotations for yolov3 in format: label_idx x_center y_center width height
    x_anna = np.array(meta_data['x_tick_window']).reshape((len(meta_data['x_tick_window']),4))
    x_anna = np.c_[np.zeros(len(meta_data['x_tick_window'])),x_anna] 

    y_anna = np.array(meta_data['y_tick_window']).reshape((len(meta_data['y_tick_window']),4))
    y_anna = np.c_[np.ones(len(meta_data['y_tick_window'])),y_anna]	
    anna = np.r_[x_anna, y_anna]
	#need to normalize the bounding box to the file size
    height_px, width_px = fig.get_size_inches()*fig.dpi

    annote_transform = lambda x: (x[0], (x[1]+x[3]/2-x[1]/2)/height_px, 
                                  1-(x[2]+x[4]/2-x[2]/2)/width_px, (x[3]-x[1])/height_px, (x[4]-x[2])/width_px)
    anna_yolo = np.apply_along_axis(annote_transform,1, anna)
    np.savetxt(name+'.txt', anna_yolo, delimiter=' ')
    plt.close('all')
    return meta_data

if __name__ == '__main__':
    test=rand_plot('main')

