class LikesController < ApplicationController
  def create
    @like = current_user.likes.create(hang_out_id: params[:hang_out_id])
    @hang_out_id = @like.hang_out_id
    redirect_to "http://localhost:3000/hang_outs/#{@hang_out_id}/confirm"
  end

  def destroy
    @like = Like.find_by(hang_out_id: params[:hang_out_id], user_id: current_user.id)
    @hang_out_id = @like.hang_out_id
    @like.destroy
    redirect_to "http://localhost:3000/hang_outs/#{@hang_out_id}/confirm_destroy"
  end
end
