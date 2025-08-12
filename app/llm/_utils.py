from jinja2 import Template

def render_prompt(template_path, docs, question, chat_history):
    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()
    template = Template(template_str)
    return template.render(docs=docs, question=question, chat_history=chat_history)

