import os

docs_dir = r'c:\Users\Pedro\My Drive\centrale-supelec\pole-anomaly\documentation'

def split_file(file_name, theory_name, impl_name, split_marker):
    path = os.path.join(docs_dir, file_name)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parts = content.split(split_marker)
    if len(parts) == 2:
        theory_content = parts[0].strip() + '\n'
        impl_content = split_marker + parts[1]
        
        with open(os.path.join(docs_dir, theory_name), 'w', encoding='utf-8') as f:
            f.write(theory_content)
        
        with open(os.path.join(docs_dir, impl_name), 'w', encoding='utf-8') as f:
            f.write(impl_content)
        
        print(f'Successfully split {file_name}')
    else:
        print(f'Failed to split {file_name}. Parts found: {len(parts)}')

split_file('1d_cnn_implementation.tex', '1d_cnn_theory.tex', '1d_cnn_implementation.tex', '\n\\section{1D-CNN Implementation}')
split_file('resnet_implementation.tex', 'resnet_theory.tex', 'resnet_implementation.tex', '\n\\section{ResNet 1D Implementation}')
split_file('cnn_attention_implementation.tex', 'cnn_attention_theory.tex', 'cnn_attention_implementation.tex', '\n\\section{CNN+Attention Implementation}')
