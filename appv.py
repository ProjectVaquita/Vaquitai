# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:39:47
:LastEditTime: 2023-02-18 23:44:52
:Description: 
"""
import PIL
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import yaml
import data_juicer.platform.src.pages.doc
import data_juicer.platform.src.pages.home
import data_juicer.platform.src.pages.data_process
import data_juicer.platform.src.pages.data_analysis

# Read page configuration
favicon = PIL.Image.open('./data_juicer/platform/src/assets/favicon.png')
st.set_page_config(
    page_title='Vaquitai',
    page_icon=favicon,
    layout='wide',
    initial_sidebar_state='auto'
)

# Read authentication configuration
with open('configs/authenticator.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


def main():
    # Define application list
    apps = [
        {"func": data_juicer.platform.src.pages.home, "title": "首页", "icon": "house"},
        {"func": data_juicer.platform.src.pages.doc, "title": "文档", "icon": "book"},
        {"func": data_juicer.platform.src.pages.data_process, "title": "运行", "icon": "caret-right-square"}, 
        {"func": data_juicer.platform.src.pages.data_analysis, "title": "结果", "icon": "bar-chart-line"},
        {"func": data_juicer.platform.src.pages.report, "title": "报告", "icon": "airplane-engines"}
    ]
    

    titles = [app["title"] for app in apps]
    titles_lower = [title.lower() for title in titles]
    icons = [app["icon"] for app in apps]

    params = st.query_params
    default_index = titles_lower.index(params.get("page", [titles[0].lower()])[0].lower())

    with st.sidebar:
        selected = option_menu(
            "Vaquitai",
            options=titles,
            icons=icons,
            menu_icon="box",
            default_index=default_index,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "20px"},
                "nav-link": {"font-size": "17px", "text-align": "left", "margin": "2px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#24A608"}
            }
        )
        st.title("About")
        st.image("https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/qrcode.jpg")
        st.info(
        """
        This web [app](http://datacentric.club/) is maintained by [vaquitai](https://github.com/ProjectVaquitai). You can follow us on community:
            [GitHub](https://github.com/ProjectVaquitai/Vaquitai),
            [小红书](https://www.xiaohongshu.com/user/profile/5cb921b0000000001102f005?xhsshare=CopyLink&appuid=63516e68000000001901e7dc&apptime=1709825164),
            [公众号](https://mp.weixin.qq.com/s/8ZQnHgp07rvTGEluoKp-rw),
            [知乎](https:///www.zhihu.com/people/VaquitaAI),
        
        """
        )
        st.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout()

    for app in apps:
        if app["title"] == selected:
            app["func"].write()
            break   

if __name__ == '__main__':
    # Authentication
    name, authentication_status, username = authenticator.login(
        fields={
            'Form name': 'Login-AI数据智能处理平台',
            'Username': 'Username',
            'Password': 'Password',
            'Login': 'Login'
        }
    )
    
    # Execute the main program based on the authentication status
    if st.session_state["authentication_status"]:
        main()
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')