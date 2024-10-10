def tag_speaker(utterances):
    formatted_dialogue = ""
    current_speaker = None

    for utterance in utterances:
        speaker = utterance['spk']  # 화자 정보 (0: 약사, 1: 환자 등으로 가정)

        # 화자가 바뀌면 화자 정보를 추가
        if speaker != current_speaker:
            current_speaker = speaker
            if current_speaker == 0:
                formatted_dialogue += "\n[약사]: "
            elif current_speaker == 1:
                formatted_dialogue += "\n[환자]: "

        # 현재 화자의 발화 추가
        formatted_dialogue += f"{utterance['msg']} "

    return formatted_dialogue.strip()
