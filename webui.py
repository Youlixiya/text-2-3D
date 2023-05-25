import gradio as gr
from sd import (load_img2img_pipeline,
                clear_img2img_pipeline,
                load_text2img_pipeline,
                clear_text2img_pipeline,
                sd_img2img,
                sd_text2img)
from blip import (load_blip,
                  clear_blip,
                  blip_generate_caption)
from sam import (load_segment_anything,
                 clear_sam,
                 add_points_boxes,segment)

def main():
    with gr.Blocks() as demo:
        with gr.Tab("Segment-anything"):
            with gr.Row():
                sam_input_image = gr.Image(label='sam_input_image')
                sam_add_point_boxes_image = gr.Image(label='sam_add_point_boxes_image')
                segment_image = gr.Image(label='segment_image')
                segment_object = gr.Image(label='segment_object')
            with gr.Row():
                sam_checkpoint = gr.Dropdown(["sam_vit_h_4b8939.pth", "sam_vit_l_0b3195.pth", "sam_vit_b_01ec64.pth"], label="sam_checkpoint")
                sam_model_type = gr.Dropdown(["vit_h", "vit_l", "vit_b"], label="sam_model_type")
                load_sam_but = gr.Button("load_sam")
                load_sam_text = gr.Textbox(label='sam_load')
                clear_sam_but = gr.Button("clear_sam")
                clear_sam_text = gr.Textbox(label='clear_sam')
            with gr.Row():
                sam_input_image_name = gr.Textbox(label="sam_input_image_name")
                sam_input_point_x = gr.Textbox(label="sam_input_point_x")
                sam_input_point_y= gr.Textbox(label="sam_input_point_y")
                sam_input_label = gr.Textbox(label="sam_input_label")
            with gr.Row():
                sam_input_boxes_x1 = gr.Textbox(label="sam_input_boxes_x1")
                sam_input_boxes_y1= gr.Textbox(label="sam_input_boxes_y1")
                sam_input_boxes_x2 = gr.Textbox(label="sam_input_boxes_x2")
                sam_input_boxes_y2= gr.Textbox(label="sam_input_boxes_y2")
            sam_add_points_boxes_but = gr.Button("sam_add_points_boxes")
            with gr.Row():
                segment_dropdown = gr.Dropdown(["segment_0", "segment_1", "segment_2"], label="segment_index")
                return_object = gr.Dropdown(['True', 'False'], label="return_object")
            with gr.Row():
                segment_but = gr.Button("segment")
                object_but = gr.Button("get_object")
            load_sam_but.click(fn=load_segment_anything, inputs=[sam_checkpoint, sam_model_type], outputs=load_sam_text)
            clear_sam_but.click(fn=clear_sam, inputs=None, outputs=clear_sam_text)
            segment_but.click(fn=segment, inputs=[sam_input_image, sam_input_image_name, sam_input_point_x, sam_input_point_y, sam_input_label, sam_input_boxes_x1, sam_input_boxes_y1, sam_input_boxes_x2, sam_input_boxes_y2, segment_dropdown], outputs=segment_image)
            sam_add_points_boxes_but.click(fn=add_points_boxes, inputs=[sam_input_image, sam_input_image_name, sam_input_point_x, sam_input_point_y, sam_input_label, sam_input_boxes_x1, sam_input_boxes_y1, sam_input_boxes_x2, sam_input_boxes_y2], outputs=sam_add_point_boxes_image)
            object_but.click(fn=segment, inputs=[sam_input_image, sam_input_image_name, sam_input_point_x, sam_input_point_y, sam_input_label, sam_input_boxes_x1, sam_input_boxes_y1, sam_input_boxes_x2, sam_input_boxes_y2, segment_dropdown, return_object], outputs=segment_object)
        with gr.Tab("BLIP"):
            with gr.Row():
                blip_input_image = gr.Image(label='blip_input_image', type='pil')
                blip_caption = gr.Textbox(label="blip_caption")
            with gr.Row():
                blip_model_key = gr.Dropdown(["Salesforce/blip-image-captioning-base"], label="blip_model_key")
                load_blip_but = gr.Button("load_blip")
                load_blip_text = gr.Textbox(label='blip_load')
            with gr.Row():
                blip = gr.Button("blip_generate_caption")
                clear_blip_but = gr.Button("clear_blip")
                clear_blip_text = gr.Textbox(label='clear_blip')
            load_blip_but.click(fn=load_blip, inputs=[blip_model_key], outputs=load_blip_text)
            blip.click(fn=blip_generate_caption, inputs=[blip_input_image], outputs=blip_caption)
            clear_blip_but.click(fn=clear_blip, inputs=None, outputs=clear_blip_text)
        with gr.Tab("Stable_diffusion"):
            with gr.Row():
                sd_prompt = gr.Textbox(label="prompt")
                sd_guidance_image = gr.Image(label='sd_guidance_image', type='pil')
                sd_generate_image = gr.Image(label='sd_generate_image', type='pil')
            with gr.Row():
                sd_model_key = gr.Dropdown(["runwayml/stable-diffusion-v1-5"], label="sd_model_key")
                load_stable_diffusion_img2img_pipeline_but = gr.Button("load_stable_diffusion_img2img_pipeline")
                load_stable_diffusion_text2img_pipeline_but = gr.Button("load_stable_diffusion_text2img_pipeline")
                load_stable_diffusion_text = gr.Textbox(label='stable_diffusion_pipeline_load')
            with gr.Row():
                clear_stable_diffusion_img2img_pipeline_but = gr.Button("clear_stable_diffusion_img2img_pipeline")
                clear_stable_diffusion_text2img_pipeline_but = gr.Button("clear_stable_diffusion_text2img_pipeline")
                clear_stable_diffusion_text = gr.Textbox(label='clear_stable_diffusion_pipeline')
            with gr.Row():
                sd_img2img_save_name = gr.Textbox(label="sd_save_name")
                strength = gr.Textbox(label="strength", value=0.8)
                guidance_scale = gr.Textbox(label='guidance_scale', value=7.5)
                sd_img2img_but = gr.Button("stable_diffusion_img2img")
            with gr.Row():
                sd_text2img_save_name = gr.Textbox(label="sd_save_name")
                sd_text2img_but = gr.Button("stable_diffusion_text2img")
            load_stable_diffusion_img2img_pipeline_but.click(fn=load_img2img_pipeline, inputs=[sd_model_key], outputs=load_stable_diffusion_text)
            load_stable_diffusion_text2img_pipeline_but.click(fn=load_text2img_pipeline, inputs=[sd_model_key], outputs=load_stable_diffusion_text)
            sd_img2img_but.click(fn=sd_img2img, inputs=[sd_guidance_image, sd_prompt, sd_img2img_save_name, strength, guidance_scale], outputs=sd_generate_image)
            sd_text2img_but.click(fn=sd_text2img, inputs=[sd_prompt, sd_text2img_save_name], outputs=sd_generate_image)
            clear_stable_diffusion_img2img_pipeline_but.click(fn=clear_img2img_pipeline, inputs=None, outputs=clear_stable_diffusion_text)
            clear_stable_diffusion_text2img_pipeline_but.click(fn=clear_text2img_pipeline, inputs=None, outputs=clear_stable_diffusion_text)
    demo.launch(share=True)
if __name__ == '__main__':
    main()
