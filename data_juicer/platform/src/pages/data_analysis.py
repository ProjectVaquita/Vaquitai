# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:58:23
:LastEditTime: 2023-02-19 14:01:40
:Description: 
"""
# Standard library
import os
import shutil
import base64
import time
import numpy as np
import pandas as pd
import faiss
import altair as alt
from loguru import logger
import plotly.graph_objects as go
import torch.nn.functional as F
import streamlit as st
import sweetviz as sv
import extra_streamlit_components as stx
import streamlit.components.v1 as components
from pathlib import Path
from PIL import Image
from data_juicer.format.load import load_formatter
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.vis import plot_dups
from data_juicer.platform.src.utils.st_components import get_remote_ip
import random


issue_dict = {'重复': '__dj__is_image_duplicated_issue',
                '低信息': '__dj__is_low_information_issue',                 
                '特殊大小': '__dj__is_odd_size_issue', 
                '特殊尺寸': '__dj__is_odd_aspect_ratio_issue',
                '极亮': '__dj__is_light_issue',
                '灰度': '__dj__is_grayscale_issue', 
                '极暗': '__dj__is_dark_issue', 
                '模糊': '__dj__is_blurry_issue'}

@st.cache_resource
def load_model():
    model_key = prepare_model(model_type='hf_blip', model_key='Salesforce/blip-itm-base-coco')
    model, processor = get_model(model_key)
    return model, processor

@st.cache_resource
def load_dataset(data_path):
    formatter = load_formatter(data_path)
    processed_dataset = formatter.load_dataset(4)
    return processed_dataset

@st.cache_resource
def create_faiss_index(emb_list):
    image_embeddings = np.array(emb_list).astype('float32')
    faiss_index = faiss.IndexFlatL2(image_embeddings.shape[1])
    faiss_index.add(image_embeddings)
    return faiss_index

def display_dataset(dataset, cond, show_num, desp, type, all=True):
    # examples = dataframe.loc[cond]
    # if all or len(examples) > 0:
    #     st.subheader(
    #         f'{desp}: :red[{len(examples)}] of '
    #         f'{len(dataframe.index)} {type} '
    #         f'(:red[{len(examples)/len(dataframe.index) * 100:.2f}%])')

    #     st.dataframe(examples[:show_num], use_container_width=True)
    
    examples = dataset.select(np.where(cond)[0])
    # examples = dataset.loc[cond]
    if all or len(examples) > 0:
        st.subheader(
            f'{desp}: :red[{len(examples)}] of '
            f'{len(dataset)} {type} '
            f'(:red[{len(examples)/len(dataset) * 100:.2f}%])')
        dataframe = pd.DataFrame(examples)
        st.write(dataframe[:show_num])


def display_image_grid(urls, cols=3, width=300):
    num_rows = int(np.ceil(len(urls) / cols))
    image_grid = np.zeros((num_rows * width, cols * width, 3), dtype=np.uint8)

    for i, url in enumerate(urls):
        image = Image.open(url)
        image = image.resize((width, width))
        image = np.array(image)
        row = i // cols
        col = i % cols
        image_grid[row * width:(row + 1) * width, col * width:(col + 1) * width, :] = image

    st.image(image_grid, channels="RGB")


# @st.cache_data
def convert_to_jsonl(df):
    return df.to_json(orient='records', lines=True, force_ascii=False).encode('utf_8_sig')


def plot_image_clusters(dataset):
    __dj__image_embedding_2d = np.array(dataset['__dj__image_embedding_2d'])
    df = pd.DataFrame(__dj__image_embedding_2d, columns=['x', 'y'])
    df['image'] = dataset['image']
    df['image_path'] = dataset['image']
    df['description'] = dataset['__dj__image_caption']
    retained_flag = np.full((len(dataset['image']), ), True, dtype=bool)
    for key, issue in issue_dict.items():
        if issue not in dataset.features:
            continue
        retained_flag = retained_flag & ~np.array(dataset[issue])
    df['category'] = ['retained' if flag else 'discarded' for flag in retained_flag]
    color_scale = alt.Scale(domain=['retained', 'discarded'],
                            range=['green', 'red'])
    brush = alt.selection_interval(encodings=['x', 'y'],
                               bind='scales'
                              )

    marker_chart = alt.Chart(df).mark_circle().encode(
        x=alt.X('x', scale=alt.Scale(type='linear', domain=[df['x'].min() * 0.95, df['x'].max() * 1.05]), axis=alt.Axis(title='X-axis')),
        y=alt.Y('y', scale=alt.Scale(type='linear', domain=[df['y'].min() * 0.95, df['y'].max() * 1.05]), axis=alt.Axis(title='Y-axis')),
        href=('image:N'), 
        tooltip=['image', 'image_path', 'description'],
        color=alt.Color('category', scale=color_scale),
    ).properties(
        width=800,
        height=600,
    ).configure_legend(
        disable=False
    ).add_params(brush).properties(width=800, height=600)
    return marker_chart

def write():
    chosen_id = stx.tab_bar(data=[
                    stx.TabBarItemData(id="data_show", title="数据展示", description=""),
                    stx.TabBarItemData(id="data_cleaning", title="数据清洗", description=""),
                    stx.TabBarItemData(id="data_mining", title="数据挖掘", description=""),
                    stx.TabBarItemData(id="data_insights", title="数据洞察", description=""),
                ], default="data_show")

    try:
        processed_dataset = load_dataset('./outputs/demo-vaquitai/demo-processed.jsonl')  
        # processed_dataset = pd.DataFrame(processed_dataset)
    except:
        st.warning('请先执行数据处理流程 !')
        st.stop()

    # TODO: Automatically find data source
    # data_source = {'BDD100K-train': 'train', 'BDD100K-val': 'val', 'BDD100K-test': 'test'}
    data_source = {'train': 'train'}
    
    def find_key_by_value(dictionary, target_value):
        print(dictionary, target_value)
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None

    if chosen_id == 'data_show':
        logger.info(f"enter data_show page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
        if '__dj__image_embedding' in processed_dataset.features:
            df = processed_dataset.remove_columns(['__dj__image_embedding']).flatten().to_pandas()
        else:
            df = processed_dataset.flatten().to_pandas()
        st.dataframe(df)
        

    if chosen_id == 'data_cleaning':
        logger.info(f"enter data_cleaning page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
        t0 = time.time()
        # dc_df = processed_dataset.remove_columns(["attributes", "labels"])
        dc_df = processed_dataset
        # category = st.selectbox("选择数据类型", list(data_source.keys()))
        # dc_df = dc_df.filter(lambda example: example['data_source'] == data_source[category])
        filter_nums = {}
        # iterate over the dataset to count the number of samples that are discarded
        all_conds = np.ones(len(dc_df['image']), dtype=bool)
        print("ori all conds: ", all_conds)
        for key in dc_df.features:
            if 'issue' not in key:
                continue
            all_conds = all_conds & (np.array(dc_df[key]) == False)
            filter_nums[key] = sum(np.array(dc_df[key]) == True)
        print('----------------------',time.time() - t0)
        @st.cache_data
        @st.cache_resource
        def draw_sankey_diagram(source_data, target_data, value_data, labels):
            """https://plotly.com/python/sankey-diagram/"""
            print(source_data)
            print(target_data)
            print(value_data)
            print(labels)
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=labels
                ),
                link=dict(
                    source=source_data,
                    target=target_data,
                    value=value_data
                )
            )])
            fig.update_layout(title_text="数据清洗比例统计", title_font=dict(size=25), font_size=16)
            st.plotly_chart(fig)

        cnt = 2
        source_data = [0, 0]
        target_data = [cnt - 1, cnt]
        # value_data = [1 - sum(filter_nums.values()) / len(dc_df['image'])]
        value_data = [sum(all_conds) / len(dc_df['image'])]
        value_data.append(1 - value_data[0])
        labels = ['原始数据', '保留: ' + str(round(value_data[0]*100, 2)) + '%', '问题数据: ' + str(round(value_data[1]*100, 2)) + '%']
        for key, value in filter_nums.items():
            if value == 0:
                continue
            cnt += 1
            source_data.append(2)
            target_data.append(cnt)
            value_data.append(value/len(dc_df[key]))
            labels.append(find_key_by_value(issue_dict, key) + ": " + str(round(value_data[-1]*100, 2)) + '%')
        
        draw_sankey_diagram(source_data, target_data, value_data, labels)
        
        cat_issue_dict = {}
        for key, value in issue_dict.items():
            res_tmp = filter_nums.get(value, None)
            if res_tmp:
                if res_tmp > 0:
                    cat_issue_dict[key] = value
        
        images_per_col = 3
        st.text("")
        st.markdown("""
                    <style>
                    .big-font {
                        font-size:25px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">数据清洗结果展示</p>', unsafe_allow_html=True)
        category_issue = st.selectbox("选择错误类型", list(cat_issue_dict.keys()))
        # amount = st.slider("展示数量", min_value=1, max_value=10, value=3, step=1)
        amount = 3
        if category_issue:
            logger.info(f"click clean_sample_show button, {category_issue}, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
            # selected_issues = dc_df[dc_df[issue_dict[category_issue]] == True]
            selected_issues = dc_df.filter(lambda example: example[issue_dict[category_issue]] == True)
            # selected_rows = selected_issues.sample(min(amount, len(selected_issues)))
            # selected_rows = selected_issues.sample(seed=42).select([0, 1, 2, 3, 4])
            selected_rows = selected_issues.shuffle()[:amount]
            if category_issue != '重复':
                random_images = selected_rows['image']
                for i in range(0, len(random_images), images_per_col):
                    cols = st.columns(images_per_col)
                    for col, img_url in zip(cols, random_images[i:i+images_per_col]):
                        col.image(img_url, use_column_width=True)
            else:
                ori_images = selected_rows['image']
                dup_images = selected_rows['__dj__duplicated_pairs']
                for i in range(0, len(ori_images), images_per_col):
                    cols = st.columns(images_per_col)
                    for col, ori_img, dup_imgs_all in zip(cols, ori_images[i:i+images_per_col], dup_images[i:i+images_per_col]):
                        dup_imgs = random.sample(dup_imgs_all, min(len(dup_imgs_all), 12))
                        display_image = plot_dup_images(ori_img, dup_imgs, len(dup_imgs_all))
                        col.pyplot(display_image)              
        
        print(all_conds)
        display_dataset(dc_df, all_conds, 10, 'Retained sampels', 'images')
        import json
        st.download_button('Download Retained data as JSONL',
                           data=convert_to_jsonl(dc_df.select(np.where(all_conds)[0]).to_pandas()),
                           file_name='retained.jsonl')
        display_dataset(dc_df, np.invert(all_conds), 10, 'Discarded sampels', 'images')
        st.download_button('Download Discarded data as JSONL',
                           data=convert_to_jsonl(dc_df.select(np.where(np.invert(all_conds))[0]).to_pandas()),
                           file_name='discarded.jsonl')

    elif chosen_id == 'data_mining':
        logger.info(f"enter data_mining page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")

        if '__dj__image_embedding_2d' not in processed_dataset.features:
            st.warning('请先执行数据处理流程(加入特征提取的算子) !')
            st.stop()

        faiss_index = create_faiss_index(processed_dataset['__dj__image_embedding'])
        model, processor = load_model()

        # 用户输入文本框
        input_text = st.text_input("", 'a picture of horse')

        # 搜索按钮
        search_button = st.button("搜索", type="primary", use_container_width=True)

        if search_button:
            inputs = processor(text=input_text, return_tensors="pt")
            text_output = model.text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True) 
            text_feature = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1).detach().cpu().numpy() 

            D, I = faiss_index.search(text_feature.astype('float32'), 10)
            retrieval_image_list = [processed_dataset['image'][i] for i in I[0]]
            # display_image_grid(retrieval_image_list, 5, 300)
            # Display the retrieved images using st.image
            for idx, image_path in enumerate(retrieval_image_list):
                st.image(image_path, caption='Path: ' + image_path + '  ,Distance: ' + str(D[0][idx]), use_column_width=False)

    elif chosen_id == 'data_insights':
        logger.info(f"enter data_insights page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
        col1, col2, col3 = st.columns(3)
        compare_features = list(processed_dataset.features)

        with col1:
            category_1 = st.selectbox('选择数据集1', list(data_source.keys()))

        with col2:
            category_2 = st.selectbox('选择数据集2', ['None'])

        with col3:
            st.write(' ')
            analysis_button = st.button("开始分析数据", type="primary", use_container_width=False)

        # dc_df = processed_dataset.filter(lambda example: example['data_source'] == data_source[category_1])
        df1 = processed_dataset.flatten().to_pandas()[compare_features]
        array_columns = df1.select_dtypes(include=[np.ndarray]).columns
        df1 = df1.drop(columns=array_columns)

        if category_2 != 'None':
            # df2 = processed_dataset.to_pandas()[compare_features]
            dc_df = processed_dataset.filter(lambda example: example['data_source'] == data_source[category_2])
            df2 = dc_df.flatten().to_pandas()[compare_features]

       
        if analysis_button:
            logger.info(f"click analysis button, {category_1}, {category_2}, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
            # st.markdown('<iframe src="http://datacentric.club:3000/" width="600" height="500"></iframe>', unsafe_allow_html=True)
            if '__dj__image_embedding_2d'  in processed_dataset.features:
                st.markdown("<h1 style='text-align: center; font-size:25px; color: black;'>数据分布可视化", unsafe_allow_html=True)
                plot = plot_image_clusters(processed_dataset)
                st.altair_chart(plot)
            else:
                st.warning('请先执行数据处理流程(加入特征提取的算子) !')

            html_save_path = os.path.join('frontend', st.session_state['username'], \
                                          category_1 + '_vs_' + category_2 + '_EDA.html')
            shutil.os.makedirs(Path(html_save_path).parent, exist_ok=True)
            with st.expander('数据集对比分析', expanded=True):
                if not os.path.exists(html_save_path ):
                    with st.spinner('Wait for process...'):
                        if category_2 == 'None':
                            report = sv.analyze(df1)
                        else:
                            report = sv.compare(df1, df2)
                    report.show_html(filepath=html_save_path, open_browser=False, layout='vertical', scale=1.0)
                components.html(open(html_save_path).read(), width=1100, height=1200, scrolling=True)