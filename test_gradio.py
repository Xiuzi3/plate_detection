import gradio as gr

def greet(name):
    return f"Hello {name}!"

# 创建一个简单的Gradio界面来测试是否还有Pydantic错误
demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Enter your name"),
    outputs=gr.Textbox(label="Greeting"),
    title="Test Gradio Interface"
)

if __name__ == "__main__":
    print("Testing Gradio interface...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
