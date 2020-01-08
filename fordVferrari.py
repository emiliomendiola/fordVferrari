#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Import necessary libraries
from fastai.vision import *
from fastai.widgets import *


# ### Download images
# The CSV files with image URLs is now uploaded to data/fordVferrari

# In[5]:


folders = ['ford', 'ferrari']

for folder in folders:
    path = Path('data/fordVferrari')
    dest = path/folder
    dest.mkdir(parents=True, exist_ok=True)
    
    file = folder+'.csv'
    download_images(path/file, dest, max_pics=200)
    
path.ls()


# Remove images that can't be opened:

# In[6]:


classes = ['ford', 'ferrari']
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# ### View data

# In[7]:


# Create data bunch
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[8]:


data.classes


# In[9]:


data.show_batch(rows=3, figsize=(7,8))


# In[10]:


# Print classes, number of classes, number of photos in training set and validation set
data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ### Train model

# In[13]:


# Create convolutional neural network (CNN)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[14]:


learn.fit_one_cycle(4)


# In[15]:


learn.save('stage-1')


# In[16]:


learn.unfreeze()


# In[17]:


learn.lr_find()


# In[18]:


learn.recorder.plot()


# In[19]:


learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))


# In[20]:


learn.save('stage-2')


# ### Interpretation

# In[21]:


learn.load('stage-2');


# In[22]:


interp = ClassificationInterpretation.from_learner(learn)


# In[23]:


interp.plot_confusion_matrix()


# ### Clean data

# In[24]:


db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )


# In[25]:


learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');


# In[26]:


ds, idxs = DatasetFormatter().from_toplosses(learn_cln)


# In[27]:


# Don't run this in google colab or any other instances running jupyter lab.
# If you do run this on Jupyter Lab, you need to restart your runtime and
# runtime state including all local variables will be lost.
ImageCleaner(ds, idxs, path)


# In[28]:


db = (ImageList.from_csv(path, 'cleaned.csv', folder='.')
                    .split_none()
                    .label_from_df()
                    .transform(get_transforms(), size=224)
                    .databunch()
      )


# In[29]:


learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');

ds, idxs = DatasetFormatter().from_similars(learn_cln)


# In[30]:


ImageCleaner(ds, idxs, path, duplicates=True)


# ### Retrain model

# In[31]:


np.random.seed(42)
data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[32]:


# Create convolutional neural network (CNN)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(5)


# In[33]:


learn.save('stage-3')


# In[34]:


learn.unfreeze()
learn.lr_find()


# In[35]:


learn.recorder.plot()


# In[38]:


learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))


# In[39]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[40]:


learn.save('stage-4')


# ### Production

# In[41]:


learn.export()


# In[42]:


path


# In[45]:


img = open_image(path/'ferrari'/'00000126.jpg')
img


# In[46]:


learn = load_learner(path)

pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:




