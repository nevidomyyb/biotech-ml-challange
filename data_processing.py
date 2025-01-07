import pickle
import pandas as pd


def create_dataframe():
    with open('./mini_gm_public_v0.1.p', 'rb') as p_file:
        data = pickle.load(p_file)
    rows = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                rows.append({
                    "syndrome_id": syndrome_id,
                    "subject_id": subject_id,
                    "image_id": image_id,
                    "embedding": embedding
                })
    df = pd.DataFrame(rows)
    df = df.dropna() 
    return df

def data_processing():
    dataframe = create_dataframe()
    number_of_syndromes = dataframe["syndrome_id"].nunique()
    df_group_syndrome = dataframe.groupby("syndrome_id").size().reset_index(name="images_count")
    total_images = df_group_syndrome["images_count"].sum()
    df_group_syndrome['images_count_percent'] = (df_group_syndrome['images_count']/total_images) * 100
    mean_value_images = df_group_syndrome["images_count"].mean()
    print(f"Number of uniques syndromes: {number_of_syndromes}")
    print(f"Total number of images: {total_images}")
    print(f"Average of images: {mean_value_images}")
    print(df_group_syndrome)
    
if __name__ == "__main__":
    data_processing()
    