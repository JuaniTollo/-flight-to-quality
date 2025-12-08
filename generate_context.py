import os

def collect_code(output_file='contexto_repo_limpio.txt', extensions=['.py', '.sh', '.md', '.ipynb', '.yaml']):
    # Lista explícita de carpetas a IGNORAR para evitar leer librerías
    ignore_dirs = {
        '.venv', 'venv', 'env', '.git', '__pycache__', '.idea', '.vscode', 
        'node_modules', 'build', 'dist', 'site-packages', 'lib', 'bin', 'include'
    }
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Recorrer el directorio actual
        for root, dirs, files in os.walk('.'):
            # Filtrar carpetas ignoradas 'in-place' para que os.walk no entre en ellas
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    
                    # Ignorar este mismo script y el archivo de salida
                    if file == os.path.basename(__file__) or file == output_file:
                        continue
                        
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                            outfile.write(f"\n{'='*40}\n")
                            outfile.write(f"FILE: {file_path}\n")
                            outfile.write(f"{'='*40}\n\n")
                            outfile.write(content)
                            outfile.write("\n")
                            
                        print(f"Agregado: {file_path}")
                        
                    except Exception as e:
                        print(f"Error leyendo {file_path}: {e}")

    print(f"\n¡Listo! Todo tu código (sin librerías) se guardó en: {output_file}")

if __name__ == "__main__":
    collect_code()