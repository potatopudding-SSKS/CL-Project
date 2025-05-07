from googleapiclient.discovery import build

all_comments = []
video_id = ""
api_key = 'AIzaSyCk84uaYs0NkkRpmFrKaWYmBR6_UIszRF4'

# recursive function to get all comments
def get_comments(youtube, video_id, next_view_token):
    global all_comments

    # check for token
    if len(next_view_token.strip()) == 0:
        all_comments = []

    if next_view_token == '':
        # get the initial response
        comment_list = youtube.commentThreads().list(part = 'snippet', maxResults = 100, videoId = video_id, order = 'relevance').execute()
    else:
        # get the next page response
        comment_list = youtube.commentThreads().list(part = 'snippet', maxResults = 100, videoId = video_id, order='relevance', pageToken=next_view_token).execute()
    # loop through all top level comments
    for comment in comment_list['items']:
        # add comment to list
        all_comments.append([comment['snippet']['topLevelComment']['snippet']['textDisplay']])
        # get number of replies
        reply_count = comment['snippet']['totalReplyCount']
        all_replies = []
        # if replies greater than 0
        if reply_count > 0:
            # get first 100 replies
            replies_list = youtube.comments().list(part='snippet', maxResults=10, parentId=comment['id']).execute()
            for reply in replies_list['items']:
                # add reply to list
                all_replies.append(reply['snippet']['textDisplay'])

        # add all replies to the comment
        all_comments[-1].append(all_replies)

    if "nextPageToken" in comment_list:
        return get_comments(youtube, video_id, comment_list['nextPageToken'])
    else:
        return []

def comment_printer():
    return all_comments


url = str(input("Enter the Youtube URL:\n"))
try:
    temp = url.split("=")
    id = temp[-1]
except:
    temp = url.split("/")
    id = temp[-1]
yt_object = build('youtube', 'v3', developerKey=api_key)
get_comments(yt_object, id, '')
comments = comment_printer()
with open(f"{id}.txt","w",encoding="utf8") as f:
    for comment in comments:
        f.write(comment[0]+"\n\n")