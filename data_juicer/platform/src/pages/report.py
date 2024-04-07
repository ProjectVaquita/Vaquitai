# -*- coding:utf-8 -*-
"""
:Date: 2023-02-19 15:05:02
:LastEditTime: 2023-02-19 15:05:04
:Description: 
"""
from loguru import logger
import streamlit as st
from data_juicer.platform.src.utils.st_components import get_remote_ip
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_juicer.utils.mm_utils import SpecialTokens
import os 

def write():
    logger.info(f"enter doc page, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
    st.title('数据集报告')
    project_path = "/home/beacon/Vaquitai/outputs/demo-vaquitai"

    # Load and calculate data
    stats = calculate_statistics(project_path)
    total_problems = sum(stats.values())

    # Display speedometer chart (you will need to use an external library for this)
    display_speedometer_chart(stats)
    display_pie_chart(stats)
    
    display_images("/home/beacon/Vaquitai/outputs/demo-vaquitai/demo-processed.jsonl")
    
    # st.markdown('<p class="big-font">数据清洗结果展示</p>', unsafe_allow_html=True)
    # category_issue = st.selectbox("选择错误类型", list(cat_issue_dict.keys()))
    # # amount = st.slider("展示数量", min_value=1, max_value=10, value=3, step=1)
    # amount = 3
    # if category_issue:
    #     logger.info(f"click clean_sample_show button, {category_issue}, user_name: {st.session_state['name']}, ip: {get_remote_ip()}")
    #     # selected_issues = dc_df[dc_df[issue_dict[category_issue]] == True]
    #     selected_issues = dc_df.filter(lambda example: example[issue_dict[category_issue]] == True)
    #     # selected_rows = selected_issues.sample(min(amount, len(selected_issues)))
    #     # selected_rows = selected_issues.sample(seed=42).select([0, 1, 2, 3, 4])
    #     selected_rows = selected_issues.shuffle()[:amount]
    #     if category_issue != '重复':
    #         random_images = selected_rows['image']
    #         for i in range(0, len(random_images), images_per_col):
    #             cols = st.columns(images_per_col)
    #             for col, img_url in zip(cols, random_images[i:i+images_per_col]):
    #                 col.image(img_url, use_column_width=True)
    #     else:
    #         ori_images = selected_rows['image']
    #         dup_images = selected_rows['__dj__duplicated_pairs']
    #         for i in range(0, len(ori_images), images_per_col):
    #             cols = st.columns(images_per_col)
    #             for col, ori_img, dup_imgs_all in zip(cols, ori_images[i:i+images_per_col], dup_images[i:i+images_per_col]):
    #                 dup_imgs = random.sample(dup_imgs_all, min(len(dup_imgs_all), 12))
    #                 display_image = plot_dup_images(ori_img, dup_imgs, len(dup_imgs_all))
    #                 col.pyplot(display_image)       
                
                
def display_pie_chart(stats):
    # Create pie chart
    tmp_stats = stats
    tmp_stats.pop("Clean")
    fig = px.pie(
        names=tmp_stats.keys(),
        values=tmp_stats.values(),
        title='问题类别图',
    )

    # Update text size
    fig.update_traces(textfont_size=20, hoverlabel_font_size=18)  # Adjust text size here
    fig.update_layout(
        autosize=False,
        width=800,  # Set the width of the chart
        height=600,  # Set the height of the chart
        legend=dict(font=dict(size=20)),  # Adjust legend font size here
        title=dict(font=dict(size=48)  # Adjust title font size here
    )
    )

    # Display pie chart
    pie_chart = st.plotly_chart(fig, use_container_width=True)
                    
    
def display_speedometer_chart(stats):
    # Create figure
    fig = go.Figure()
    score = 100 * stats["Clean"] / sum(stats.values())
    # Add trace
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "数据集得分" , 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "dodgerblue"},
            'bar': {'color': "dodgerblue"},
            'bgcolor': 'rgba(0,0,0,0)',  # Transparent background for the gauge
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 75], 'color': 'gray'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score}
        },
        number={'font': {'size': 60, 'color': "dodgerblue"}}
    ))
    
    # Add text annotation
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text="总数据量: %d <br>问题数据: %d <br>问题类别: %d" % (sum(stats.values()), sum(stats.values()) - stats["Clean"], len(stats) - 1),  # Change this to your desired text
        font=dict(size=16, color="white"),  # Customize font size and color
        showarrow=False,
        align="center"
    )
    
    # Set layout for transparency
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the paper
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background for the plot
        font={'color': "dodgerblue", 'family': "Arial"}
    )

    # Display chart
    st.plotly_chart(fig, use_container_width=True)  


# Load data function
def load_data(category):
    # Replace 'path_to_file.json' with the actual file path and ensure it is in the correct format
    with open(f'{category}.json') as f:
        data = json.load(f)
    return data

# Calculate statistics function
def calculate_statistics(project_path):
    tracer_path = "%s/trace" % project_path
    output_path = "%s/demo-processed.jsonl" % project_path
    def get_file_paths(folder_path):
        file_paths = []
        # Walk through all files and directories in the given folder
        for root, directories, files in os.walk(folder_path):
            for filename in files:
                # Join the root path and the file name to get the absolute file path
                file_paths.append(os.path.join(root, filename))
        return file_paths
    
    def get_total_line_count(jsonl_file):
        with open(jsonl_file, 'r') as file:
            total_line_count = sum(1 for line in file)
        return total_line_count
    
    def get_total_dup_nums(jsonl_file):
        total_dup_nums = 0
        with open(jsonl_file, 'r') as file:
            for line in file:
                json_obj = json.loads(line)
                dup_num = json_obj['dup_num']
                total_dup_nums += dup_num
        return total_dup_nums


    file_paths = get_file_paths(tracer_path)
    
    problems_dict, stats_dict = {}, {}
    for file_path in file_paths:
        file_p = file_path.split('/')[-1].split('.')[0]
        typ = file_p.split('-')[0]
        problem = file_p.split('-')[-1].split(".")[0]
        problems_dict[file_path] = problem
        
        if typ == "duplicate":
            stats_dict[problem] = get_total_dup_nums(file_path)
        else:
            stats_dict[problem] = get_total_line_count(file_path)
            
    stats_dict["Clean"] = get_total_line_count(output_path)
        

    return stats_dict 


def display_images(jsonl_file, num_images=5):
    st.title("Display Images with Labels")
    import jsonlines
    from PIL import Image
    # Open the JSONL file
    with jsonlines.open(jsonl_file) as reader:
        for i, item in enumerate(reader):
            if i >= num_images:
                break
            
            # Get image URL
            image_url = item["images"][0]
            
            # Load image from URL
            image = Image.open(image_url)
            
            # Display image
            st.image(image, caption="Caption: %s\n\nPath: %s" % (item["text"].split(SpecialTokens.image)[-1], image_url.split("/")[-1]))
            