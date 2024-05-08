# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:58:23
:LastEditTime: 2023-02-19 14:01:40
:Description: 
"""
# Standard library
import os
import shutil
import numpy as np
import pandas as pd
import faiss
from loguru import logger
import streamlit as st
import sweetviz as sv
import plotly.express as px
from data_juicer.utils.constant import Fields,StatsKeysConstant
import extra_streamlit_components as stx
import streamlit.components.v1 as components
from pathlib import Path
from data_juicer.format.load import load_formatter
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.platform.src.utils.st_components import get_remote_ip
from pandas import json_normalize


CAPTION_KEY = "text"
ROOT_DIR = "./outputs/demo-backbone-comb-dist-10w"
COLUMN_IMAGE_CAPTION = '.'.join([Fields.stats, StatsKeysConstant.image_caption])
COLUMN_IMAGE_EMBEDDING = '.'.join([Fields.stats, StatsKeysConstant.image_embedding])
COLUMN_IMAGE_EMBEDDING_2D = '.'.join([Fields.stats, StatsKeysConstant.image_embedding_2d])

@st.cache_resource
def load_model(model_id: str = 'Salesforce/blip-itm-base-coco'):
    model_key = prepare_model(model_type='huggingface',  pretrained_model_name_or_path=model_id)
    model, processor = get_model(model_key)
    return model, processor


@st.cache_resource
def load_jsonl2df(jsonl_path, flatten=True):
    if flatten:
        df = json_normalize(pd.read_json(jsonl_path, lines=True).to_dict('records'))
    else:
        df = pd.read_json(jsonl_path, lines=True)
    return df

@st.cache_resource
def create_faiss_index(emb_list):
    print('create_faiss_index')
    image_embeddings = np.array(emb_list).astype('float32')
    faiss_index = faiss.IndexFlatL2(image_embeddings.shape[1])
    faiss_index.add(image_embeddings)
    return faiss_index


def scatter_plot(df_dataset, x_col, y_col, color_col, hover_data, ):
    # color_list = [ 'red', 'green', 'blue', 'orange', 'purple', 'yellow']
    # category_list = list(set(df_dataset[color_col]))
    # color_discrete_map = {category_list[i]: color_list[i] for i in range(len(category_list))}
    fig = px.scatter(
        df_dataset,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=['text'],
        # color_discrete_map=color_discrete_map,
        symbol=color_col,
        opacity = 0.6,
        width=800, 
        height=800,
        size_max=10,
    )

    return fig

def parse_dups(df_dup_dataset):
    dup_list = []
    for idx, sample in df_dup_dataset.iterrows():
        for i in range(1, int(sample['dup_num']+1)):
            key = f'dup{i}'
            dup_list.append(sample[key])
    return pd.json_normalize(dup_list)

@st.cache_resource
def load_dataset(output_dir):
    # set dataset path
    processed_data_path = f'%s/demo-processed.jsonl' % output_dir
    processed_data_stats_path = f'%s/demo-processed_stats.jsonl' % output_dir
    trace_dir = '%s/trace' % output_dir
    
    # load dataset
    df_processed = load_jsonl2df(processed_data_path)
    df_stats = load_jsonl2df(processed_data_stats_path)
    df_dataset = pd.concat([df_processed, df_stats], axis=1)
    
    for path in Path(trace_dir).glob('*.jsonl'):
        if path.name.startswith('duplicate'):
            df_dup = load_jsonl2df(str(path), flatten=False)
            df_dup = parse_dups(df_dup)
            df_dataset = pd.concat([df_dataset, df_dup], axis=0)

        elif path.name.startswith('filter'):
            df_filter = load_jsonl2df(str(path))
            df_dataset = pd.concat([df_dataset, df_filter], axis=0)
        else:
            continue
    
    # dataset processed
    df_dataset['state'] = ['retained'] * len(df_processed) + ['discarded'] * (len(df_dataset) - len(df_processed))
    df_dataset = df_dataset.explode('images')
    if COLUMN_IMAGE_EMBEDDING_2D in df_dataset:
        df_dataset['emb2d_x'] = np.array(df_dataset[COLUMN_IMAGE_EMBEDDING_2D].tolist()).reshape(-1,2)[:, 0]
        df_dataset['emb2d_y'] = np.array(df_dataset[COLUMN_IMAGE_EMBEDDING_2D].tolist()).reshape(-1,2)[:, 1]
    return df_dataset



def write():
    chosen_id = stx.tab_bar(data=[
                    stx.TabBarItemData(id="data_show", title="数据展示", description=""),
                    stx.TabBarItemData(id="data_mining", title="数据挖掘", description=""),
                    stx.TabBarItemData(id="data_insights", title="数据洞察", description=""),
                ], default="data_show")

    df_dataset = load_dataset(ROOT_DIR)

    if chosen_id == 'data_show':
        logger.info(f"enter data_show page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
        if COLUMN_IMAGE_EMBEDDING in df_dataset:
            st.dataframe(df_dataset.drop(columns=[COLUMN_IMAGE_EMBEDDING]))
        else:
            st.dataframe(df_dataset)
        

    if chosen_id == 'data_cleaning':
        logger.info(f"enter data_cleaning page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
        pass
        

    elif chosen_id == 'data_mining':
        logger.info(f"enter data_mining page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
        if COLUMN_IMAGE_EMBEDDING not in df_dataset:
            st.warning('请先执行数据处理流程(加入特征提取的算子) !')
            st.stop()
        
        emb_list = np.array(df_dataset.explode(COLUMN_IMAGE_EMBEDDING)[COLUMN_IMAGE_EMBEDDING].values.tolist())
        image_list = df_dataset['images'].tolist()
        faiss_index = create_faiss_index(emb_list)
        model, processor = load_model('openai/clip-vit-base-patch32')

        # 用户输入文本框
        input_text = st.text_input("", 'a picture of horse')

        # 搜索按钮
        search_button = st.button("搜索", type="primary", use_container_width=True)

        if search_button:
            inputs = processor(text=input_text,
                    images=None,
                    return_tensors='pt',
                    truncation=True,
                    max_length=model.config.text_config.
                    max_position_embeddings,
                    padding=True).to(model.device)
            text_feature = model.get_text_features(**inputs).detach().cpu().numpy()
            D, I = faiss_index.search(text_feature.astype('float32'), 10)
            retrieval_image_list = [image_list[i] for i in I[0]]
            for idx, image_path in enumerate(retrieval_image_list):
                st.image(image_path, caption='Path: ' + image_path + '  ,Distance: ' + str(D[0][idx]), use_column_width=False)

    elif chosen_id == 'data_insights':
        logger.info(f"enter data_insights page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
        col1, col2, col3 = st.columns(3)
        select_columns = ['state', 'data_source']
        select_box = {}
        for col in select_columns:
            if col not in df_dataset:
                continue
            for val in set(df_dataset[col]):
                select_box[val] = col

        with col1:
            category_1 = st.selectbox('选择数据集1', list(select_box.keys()))

        with col2:
            category_2 = st.selectbox('选择数据集2', list(select_box.keys()))

        with col3:
            st.write(' ')
            analysis_button = st.button("开始分析数据", type="primary", use_container_width=False)

        # 提取日期和时间
        # df_dataset['timestamp'] = pd.to_datetime(df_dataset['meta_data.timestamp']//1000000, unit='s')
        # df_dataset['date'] = df_dataset['timestamp'].dt.date
        # df_dataset['time'] = df_dataset['timestamp'].dt.time

        # 显示地图
        # st.write(map)
        # fig = px.scatter_mapbox(df1, lat='meta_data.lat', lon='meta_data.lon', 
        #                         zoom=3, height=800, width=1000)
        # # fig = px.scatter_mapbox(df1, lat='meta.lat', lon='meta.lon', color='data_source',
        # #                 color_discrete_map={'train': 'blue', 'test': 'orange'},
        # #                 zoom=3, height=600)
        # fig.update_layout(mapbox_style='open-street-map')
        # st.plotly_chart(fig)
        
        df1 = df_dataset[df_dataset[select_box[category_1]] == category_1]
        df2 = df_dataset[df_dataset[select_box[category_2]] == category_2]
        pd.options.mode.chained_assignment = None
        df1['color'], df2['color'] = category_1, category_2
        df_compare = pd.concat([df1, df2], axis=0).sample(frac=1, random_state=42)

       
        if analysis_button:
            logger.info(f"click analysis button, {category_1}, {category_2}, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")    

            # 向量可视化        
            st.markdown("<h1 style='text-align: center; font-size:25px; color: black;'>数据分布可视化", unsafe_allow_html=True)

            def gen_text(image_path, caption):
                return f"image: {image_path}<br>caption: {caption}"

            df_compare['text'] = df_compare.apply(lambda row: gen_text(row['images'], row[COLUMN_IMAGE_CAPTION]), axis=1)
            
            fig = scatter_plot(df_compare, 
                               x_col='emb2d_x', 
                               y_col='emb2d_y', 
                               color_col='data_source',
                               hover_data='text')
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            # meta data
            html_save_path = os.path.join('frontend', st.session_state['username'], \
                                          category_1 + '_vs_' + category_2 + '_EDA.html')
            shutil.os.makedirs(Path(html_save_path).parent, exist_ok=True)
            normal_columns = list(set([col for col in df1.columns if not isinstance(df1[col].iloc[0], list)]) & 
                                set([col for col in df2.columns if not isinstance(df2[col].iloc[0], list)]))
            
            with st.expander('数据集对比分析', expanded=True):
                if not os.path.exists(html_save_path ):
                    with st.spinner('Wait for process...'):
                        if category_1 == category_2:
                            report = sv.analyze(df1[normal_columns])
                            # report = ProfileReport(df1[normal_columns], title="Profiling Report")
                            # report.to_file(html_save_path)
                            report.show_html(filepath=html_save_path, open_browser=False, layout='vertical', scale=1.0)
                        else:
                            report = sv.compare(df1[normal_columns], df2[normal_columns])
                            report.show_html(filepath=html_save_path, open_browser=False, layout='vertical', scale=1.0)
                components.html(open(html_save_path).read(), width=1100, height=1200, scrolling=True)
            
            # TODO: Semantics