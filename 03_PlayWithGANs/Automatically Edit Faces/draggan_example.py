from draggan.web import main as dw_main

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('-p', '--port', default=None)
    parser.add_argument('--ip', default=None)
    args = parser.parse_args()
    device = args.device
    demo = dw_main()
    print('Successfully loaded, starting gradio demo')
    demo.queue(concurrency_count=1, max_size=20).launch(share=args.share, server_name=args.ip, server_port=args.port)